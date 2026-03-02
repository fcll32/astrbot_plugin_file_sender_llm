import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.star import Context, Star


class FileSenderLLMPlugin(Star):
    """
    基于 LLM 工具调用的本地文件发送插件（白名单目录 + 安全校验 + 模糊匹配）

    安全边界：
    - 不进行任何网络请求
    - 不执行命令/危险代码（不使用 eval/exec）
    - 不做删除/篡改等破坏性文件操作
    - 仅在白名单目录内做只读检索并发送文件
    """

    # 供新版本自动发现/元信息读取（如框架支持）
    PLUGIN_NAME = "astrbot_plugin_file_sender_llm"
    PLUGIN_AUTHOR = "AstrBotPluginPlanner"
    PLUGIN_DESC = "基于 LLM 工具调用的本地文件发送插件（白名单目录 + 安全校验 + 模糊匹配）"
    PLUGIN_VERSION = "1.1.0"

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 会话待确认状态（LRU + TTL）
        self.pending_confirmations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        # 仅用于 on_llm_request 作用域约束，避免污染全局请求
        self._llm_scope_until: "OrderedDict[str, float]" = OrderedDict()

        # 后台清理任务
        self._cleanup_task: Optional[asyncio.Task] = None

        self.auto_trigger: bool = bool(self._cfg("auto_trigger", True))
        self._validate_config_contract()

        self._log(
            "info",
            "插件初始化完成，auto_trigger=%s, allowed_root_dirs_count=%d",
            self.auto_trigger,
            len(self._cfg("allowed_root_dirs", [])),
        )

    # ------------------------------
    # 配置与日志辅助
    # ------------------------------
    _TOP_LEVEL_COMPAT_MAP: Dict[str, str] = {
        # 文档顶层字段 -> 代码嵌套字段（兼容）
        "enable_fuzzy_match": "match_policy.enable_fuzzy_match",
        "max_candidates": "match_policy.max_candidates",
        "require_confirmation_on_ambiguous": "confirmation_policy.require_confirmation_on_ambiguous",
        "log_level": "logging.log_level",
    }

    def _try_get(self, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        getter = getattr(obj, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except Exception:
                return default
        return default

    def _get_nested(self, path: str) -> Any:
        curr: Any = self.config
        for key in path.split("."):
            curr = self._try_get(curr, key, None)
            if curr is None:
                return None
        return curr

    def _cfg(self, path: str, default: Any) -> Any:
        """
        读取配置（优先嵌套；兼容文档顶层字段）。
        """
        # 1) 优先读取代码中的标准嵌套路径
        v = self._get_nested(path)
        if v is not None:
            return v

        # 2) 兼容文档中的顶层旧字段
        for top_key, nested_key in self._TOP_LEVEL_COMPAT_MAP.items():
            if nested_key == path:
                tv = self._try_get(self.config, top_key, None)
                if tv is not None:
                    return tv

        # 3) path 本身就是顶层字段时直接读
        if "." not in path:
            tv = self._try_get(self.config, path, None)
            if tv is not None:
                return tv

        return default

    def _validate_config_contract(self) -> None:
        """
        启动时输出关键配置校验结果，便于发现“文档字段/实际字段”不一致。
        """
        checks = [
            ("match_policy.enable_fuzzy_match", "enable_fuzzy_match"),
            ("match_policy.max_candidates", "max_candidates"),
            ("confirmation_policy.require_confirmation_on_ambiguous", "require_confirmation_on_ambiguous"),
            ("logging.log_level", "log_level"),
        ]
        for nested, top in checks:
            nested_v = self._get_nested(nested)
            top_v = self._try_get(self.config, top, None)
            if nested_v is None and top_v is None:
                self._log("warning", "配置缺失：%s（兼容顶层 %s）未设置，使用默认值。", nested, top)
            elif nested_v is not None and top_v is not None and nested_v != top_v:
                self._log(
                    "warning",
                    "配置冲突：%s=%r 与顶层 %s=%r 不一致，已优先采用嵌套字段。",
                    nested,
                    nested_v,
                    top,
                    top_v,
                )
            else:
                effective = nested_v if nested_v is not None else top_v
                self._log("debug", "配置生效：%s (兼容 %s) => %r", nested, top, effective)

    def _log(self, level: str, msg: str, *args: Any) -> None:
        level_order = {"debug": 10, "info": 20, "warning": 30, "error": 40}
        configured = str(self._cfg("logging.log_level", "info")).lower()
        if level_order.get(level, 20) < level_order.get(configured, 20):
            return
        if level == "debug":
            logger.debug(msg, *args)
        elif level == "warning":
            logger.warning(msg, *args)
        elif level == "error":
            logger.error(msg, *args)
        else:
            logger.info(msg, *args)

    def _safe_user_text_for_log(self, text: str) -> str:
        if bool(self._cfg("logging.log_redact_user_input", True)):
            summary = text[:12] + ("..." if len(text) > 12 else "")
            return f"<redacted len={len(text)} summary={summary!r}>"
        return text

    def _display_path(self, p: Path) -> str:
        if bool(self._cfg("security_policy.redact_sensitive_path_in_reply", True)):
            return p.name
        return str(p)


    # ------------------------------
    # 文件发送权限控制
    # ------------------------------
    def _is_sender_permitted(self, event: AstrMessageEvent) -> Tuple[bool, str]:
        """
        文件发送权限控制：
        - 默认 mode=all：所有人可用
        - mode=allowlist：仅允许 allowlist_sender_ids 中的发送者使用
        """
        mode = str(self._cfg("permission_policy.mode", "all")).lower().strip()
        deny_msg = str(self._cfg("permission_policy.deny_message", "你没有权限使用文件发送功能。")).strip() or "你没有权限使用文件发送功能。"

        if mode not in {"all", "allowlist"}:
            mode = "all"

        if mode == "all":
            return True, "ok"

        allowlist = self._cfg("permission_policy.allowlist_sender_ids", [])
        if not isinstance(allowlist, list):
            allowlist = []
        allowset = {str(x).strip() for x in allowlist if str(x).strip()}

        try:
            sender_id = str(event.get_sender_id())
        except Exception:
            sender_id = ""

        if sender_id and sender_id in allowset:
            return True, "ok"

        return False, deny_msg

    # ------------------------------
    # 状态清理（TTL + 容量上限）
    # ------------------------------
    def _ensure_cleanup_task(self) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        interval = int(self._cfg("confirmation_policy.cleanup_interval_sec", 60))
        interval = max(10, interval)
        while True:
            try:
                self._prune_pending_confirmations()
                self._prune_llm_scope()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._log("warning", "后台清理任务异常: %s", str(e))
                await asyncio.sleep(interval)

    def _prune_pending_confirmations(self) -> None:
        now = time.time()
        # TTL 淘汰
        expired = [k for k, v in self.pending_confirmations.items() if now > float(v.get("expire_at", 0))]
        for k in expired:
            self.pending_confirmations.pop(k, None)

        # 容量上限淘汰（LRU）
        cap = int(self._cfg("confirmation_policy.max_pending_confirmations", 500))
        cap = max(10, cap)
        while len(self.pending_confirmations) > cap:
            self.pending_confirmations.popitem(last=False)

    def _prune_llm_scope(self) -> None:
        now = time.time()
        expired = [k for k, v in self._llm_scope_until.items() if now > float(v)]
        for k in expired:
            self._llm_scope_until.pop(k, None)

        cap = int(self._cfg("llm_tool_config.max_scoped_sessions", 500))
        cap = max(10, cap)
        while len(self._llm_scope_until) > cap:
            self._llm_scope_until.popitem(last=False)

    def _set_pending_confirmation(self, umo: str, candidates: List[Path], timeout_sec: int) -> None:
        self._prune_pending_confirmations()
        self.pending_confirmations[umo] = {
            "candidates": candidates,
            "expire_at": time.time() + timeout_sec,
        }
        self.pending_confirmations.move_to_end(umo, last=True)

    def _mark_llm_scope(self, umo: str, ttl_sec: int = 180) -> None:
        self._prune_llm_scope()
        self._llm_scope_until[umo] = time.time() + max(30, ttl_sec)
        self._llm_scope_until.move_to_end(umo, last=True)

    # ------------------------------
    # 路径安全与文件筛选
    # ------------------------------
    def _get_allowed_roots(self) -> List[Path]:
        roots_raw = self._cfg("allowed_root_dirs", [])
        roots: List[Path] = []
        if not isinstance(roots_raw, list):
            roots_raw = []
        for raw in roots_raw:
            try:
                rp = Path(str(raw)).expanduser().resolve()
                if rp.exists() and rp.is_dir():
                    roots.append(rp)
                else:
                    self._log("warning", "忽略无效白名单目录: %s", str(raw))
            except Exception:
                self._log("warning", "白名单目录解析失败: %s", str(raw))
        return roots

    @staticmethod
    def _is_subpath(child: Path, root: Path) -> bool:
        try:
            child.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False

    def _contains_parent_escape(self, text: str) -> bool:
        return ".." in text.replace("\\", "/").split("/")

    def _looks_like_abs_path(self, text: str) -> bool:
        return bool(
            re.match(r"^(/|[a-zA-Z]:\\|\\\\)", text.strip())
            or text.strip().startswith("file://")
        )

    def _is_hidden(self, p: Path) -> bool:
        return p.name.startswith(".")

    def _is_allowed_extension(self, p: Path) -> bool:
        allowed_exts = self._cfg("file_policy.allowed_file_extensions", [])
        if not isinstance(allowed_exts, list) or len(allowed_exts) == 0:
            return False
        ext = p.suffix.lower()
        normalized = {str(x).lower() for x in allowed_exts}
        return ext in normalized

    def _is_allowed_size(self, p: Path) -> bool:
        max_mb = int(self._cfg("file_policy.max_file_size_mb", 30))
        try:
            return p.stat().st_size <= max_mb * 1024 * 1024
        except Exception:
            return False

    def _check_file_security(self, p: Path, roots: List[Path]) -> Tuple[bool, str]:
        if not p.exists() or not p.is_file():
            return False, "目标文件不存在或不可读。"

        strict_boundary = bool(self._cfg("security_policy.strict_path_boundary_check", True))
        deny_symlink_outside = bool(self._cfg("security_policy.deny_symlink_target_outside_root", True))
        deny_hidden = bool(self._cfg("file_policy.deny_hidden_files", True))

        if deny_hidden and self._is_hidden(p):
            return False, "隐藏文件不允许发送。"
        if not self._is_allowed_extension(p):
            return False, "文件类型不在允许范围内。"
        if not self._is_allowed_size(p):
            return False, "文件大小超过限制。"

        if strict_boundary:
            if not any(self._is_subpath(p, root) for root in roots):
                return False, "目标文件不在白名单目录内。"

        if deny_symlink_outside and p.is_symlink():
            target = p.resolve()
            if not any(self._is_subpath(target, root) for root in roots):
                return False, "符号链接目标越界，已拒绝。"

        return True, "ok"

    # ------------------------------
    # 检索与匹配（重操作放入线程）
    # ------------------------------
    def _normalize(self, s: str) -> str:
        case_sensitive = bool(self._cfg("match_policy.case_sensitive_match", False))
        return s if case_sensitive else s.lower()

    def _score_match(self, query: str, filename: str, stem: str) -> float:
        q = self._normalize(query.strip())
        fn = self._normalize(filename)
        st = self._normalize(stem)

        if not q:
            return 0.0
        if q == fn or q == st:
            return 1.0
        if q in fn or q in st:
            return 0.95
        return max(SequenceMatcher(None, q, fn).ratio(), SequenceMatcher(None, q, st).ratio())

    def _iter_files_under_root(self, root: Path) -> List[Path]:
        include_sub = bool(self._cfg("match_policy.include_subdirectories", True))
        files: List[Path] = []

        if include_sub:
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    files.append(Path(dirpath) / name)
        else:
            try:
                for item in root.iterdir():
                    if item.is_file():
                        files.append(item)
            except Exception:
                self._log("warning", "扫描目录失败: %s", str(root))
        return files

    def _search_candidates_sync(self, query: str, roots: List[Path]) -> List[Tuple[Path, float, float]]:
        enable_fuzzy = bool(self._cfg("match_policy.enable_fuzzy_match", True))
        fuzzy_min_score = float(self._cfg("match_policy.fuzzy_min_score", 0.72))
        max_candidates = int(self._cfg("match_policy.max_candidates", 5))

        results: List[Tuple[Path, float, float]] = []
        for root in roots:
            for p in self._iter_files_under_root(root):
                try:
                    if bool(self._cfg("file_policy.deny_hidden_files", True)) and self._is_hidden(p):
                        continue
                    if not self._is_allowed_extension(p):
                        continue
                    if not self._is_allowed_size(p):
                        continue
                    if bool(self._cfg("security_policy.strict_path_boundary_check", True)) and not self._is_subpath(p, root):
                        continue
                    if (
                        bool(self._cfg("security_policy.deny_symlink_target_outside_root", True))
                        and p.is_symlink()
                        and not self._is_subpath(p.resolve(), root)
                    ):
                        continue

                    score = self._score_match(query, p.name, p.stem)
                    if score >= 1.0 or (enable_fuzzy and score >= fuzzy_min_score):
                        results.append((p, score, p.stat().st_mtime))
                except Exception:
                    continue

        latest_hint = bool(self._cfg("match_policy.prefer_latest_when_query_contains_latest", True)) and any(
            k in query for k in ["最新", "最新版", "latest", "newest"]
        )

        if latest_hint:
            results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        else:
            results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_candidates]

    async def _search_candidates(self, query: str, roots: List[Path]) -> List[Tuple[Path, float, float]]:
        # 将阻塞性扫描与相似度计算放入线程，避免阻塞事件循环
        return await asyncio.to_thread(self._search_candidates_sync, query, roots)

    def _pick_from_confirmation_text(self, text: str, candidates: List[Path]) -> Optional[Path]:
        text = text.strip()
        if not text:
            return None

        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]

        norm_text = self._normalize(text)
        matched = [p for p in candidates if self._normalize(p.name) in norm_text or self._normalize(p.stem) in norm_text]
        if len(matched) == 1:
            return matched[0]

        scored = [(p, self._score_match(text, p.name, p.stem)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= 0.85:
            if len(scored) == 1 or scored[0][1] - scored[1][1] >= 0.05:
                return scored[0][0]
        return None

    async def _yield_send_file_result(self, event: AstrMessageEvent, target: Path) -> MessageEventResult:
        ok_roots = self._get_allowed_roots()
        ok, reason = self._check_file_security(target, ok_roots)
        if not ok:
            if bool(self._cfg("logging.log_failure_events", True)):
                self._log("warning", "发送前校验失败: %s", reason)
            yield event.plain_result(f"文件发送被拒绝：{reason}")
            return

        try:
            yield event.plain_result(f"正在发送文件：{self._display_path(target)}")
            yield event.chain_result([Comp.File(file=str(target), name=target.name)])

            if bool(self._cfg("logging.log_success_events", True)):
                path_for_log = target.name if bool(self._cfg("logging.log_redact_full_path", True)) else str(target)
                self._log("info", "文件发送成功: %s", path_for_log)
        except Exception as e:
            if bool(self._cfg("logging.log_failure_events", True)):
                self._log("error", "发送失败: %s", str(e))
            yield event.plain_result("文件发送失败：平台可能不支持文件消息，或文件暂不可发送。")

    # ------------------------------
    # LLM 工具：文件发送
    # ------------------------------
    @filter.llm_tool(name="send_local_file")
    async def send_local_file(self, event: AstrMessageEvent, query: str) -> MessageEventResult:
        """
        在白名单目录内安全检索并发送文件。
        Args:
            query(string): 用户自然语言中的文件关键词、文件名、版本提示等信息
        """
        self._ensure_cleanup_task()
        self._prune_pending_confirmations()

        # 权限校验：不允许的用户直接拒绝
        permitted, reason = self._is_sender_permitted(event)
        if not permitted:
            yield event.plain_result(reason)
            return

        try:
            query = (query or "").strip()
            if not query:
                yield event.plain_result("请提供要发送的文件关键词，例如：项目周报、部署说明。")
                return

            if bool(self._cfg("security_policy.deny_absolute_path_from_user", True)) and self._looks_like_abs_path(query):
                yield event.plain_result("已拒绝：不支持直接使用绝对路径，请提供文件关键词。")
                return

            if bool(self._cfg("security_policy.deny_parent_path_escape", True)) and self._contains_parent_escape(query):
                yield event.plain_result("已拒绝：检测到潜在路径越界请求。")
                return

            roots = self._get_allowed_roots()
            if not roots:
                yield event.plain_result("插件未配置可用白名单目录，请联系管理员设置 allowed_root_dirs。")
                return

            umo = event.unified_msg_origin
            pending = self.pending_confirmations.get(umo)
            if pending and time.time() < float(pending.get("expire_at", 0)):
                self.pending_confirmations.move_to_end(umo, last=True)
                selected = self._pick_from_confirmation_text(query, pending.get("candidates", []))
                if selected is not None:
                    self.pending_confirmations.pop(umo, None)
                    async for r in self._yield_send_file_result(event, selected):
                        yield r
                    return

            candidates = await self._search_candidates(query, roots)

            if not candidates:
                yield event.plain_result("未在允许目录中找到匹配文件，请补充更完整关键词。")
                return

            auto_single = bool(self._cfg("match_policy.auto_send_on_single_high_confidence", True))
            high_threshold = float(self._cfg("match_policy.high_confidence_threshold", 0.9))

            if len(candidates) == 1 and auto_single and candidates[0][1] >= high_threshold:
                async for r in self._yield_send_file_result(event, candidates[0][0]):
                    yield r
                return

            require_confirm = bool(self._cfg("confirmation_policy.require_confirmation_on_ambiguous", True))
            timeout_sec = int(self._cfg("confirmation_policy.confirmation_timeout_sec", 120))

            if len(candidates) > 1 and require_confirm:
                paths = [x[0] for x in candidates]
                self._set_pending_confirmation(umo, paths, timeout_sec)
                display = "、".join([f"{idx + 1}. {self._display_path(p)}" for idx, p in enumerate(paths)])
                yield event.plain_result(
                    f"找到多个候选文件：{display}\n请回复序号（如 1）或更具体名称进行确认。"
                )
                return

            async for r in self._yield_send_file_result(event, candidates[0][0]):
                yield r

        except Exception as e:
            self._log("error", "工具 send_local_file 异常: %s", str(e))
            yield event.plain_result("处理文件发送请求时发生异常，请稍后再试。")

    # ------------------------------
    # 自动触发：消息监听 + LLM 请求
    # ------------------------------
    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_message_auto_trigger(self, event: AstrMessageEvent):
        self._ensure_cleanup_task()
        self._prune_pending_confirmations()
        self._prune_llm_scope()

        try:
            if not self.auto_trigger:
                return

            text = (event.message_str or "").strip()
            if not text:
                return

            try:
                sender_id = str(event.get_sender_id())
                bot_id = str(event.message_obj.self_id)
                if sender_id and bot_id and sender_id == bot_id:
                    return
            except Exception:
                pass

            if text.startswith("/"):
                return

            umo = event.unified_msg_origin

            pending = self.pending_confirmations.get(umo)
            if pending:
                permitted, reason = self._is_sender_permitted(event)
                if not permitted:
                    self.pending_confirmations.pop(umo, None)
                    yield event.plain_result(reason)
                    event.stop_event()
                    return

            if pending:
                if time.time() > float(pending.get("expire_at", 0)):
                    self.pending_confirmations.pop(umo, None)
                    yield event.plain_result("文件确认已超时，请重新描述要发送的文件。")
                    event.stop_event()
                    return

                self.pending_confirmations.move_to_end(umo, last=True)
                selected = self._pick_from_confirmation_text(text, pending.get("candidates", []))
                if selected is not None:
                    self.pending_confirmations.pop(umo, None)
                    async for r in self._yield_send_file_result(event, selected):
                        yield r
                    event.stop_event()
                    return

            keywords = self._cfg(
                "llm_tool_config.intent_keywords",
                ["发我", "发一下", "传我", "发送文件", "给我文档", "把资料发来"],
            )
            if not isinstance(keywords, list):
                keywords = []

            hit_intent = any(str(k) in text for k in keywords)
            if not hit_intent:
                return

            permitted, reason = self._is_sender_permitted(event)
            if not permitted:
                yield event.plain_result(reason)
                event.stop_event()
                return

            provider = self.context.get_using_provider()
            if provider is None:
                return

            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            conversation = None
            contexts = []
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
                try:
                    contexts = json.loads(conversation.history) if conversation and conversation.history else []
                except Exception:
                    contexts = []

            # 标记当前会话，供 on_llm_request 作用域判断
            self._mark_llm_scope(umo, int(self._cfg("llm_tool_config.scope_ttl_sec", 180)))

            self._log("info", "触发自动文件意图识别: %s", self._safe_user_text_for_log(text))

            yield event.request_llm(
                prompt=text,
                func_tool_manager=self.context.get_llm_tool_manager(),
                session_id=curr_cid,
                contexts=contexts,
                system_prompt="",
                image_urls=[],
                conversation=conversation,
            )
            event.stop_event()

        except Exception as e:
            self._log("error", "自动触发处理异常: %s", str(e))

    # ------------------------------
    # LLM 请求钩子：仅限作用域内追加约束
    # ------------------------------
    def _should_inject_prompt(self, event: AstrMessageEvent) -> bool:
        """
        仅在“文件发送相关意图/本插件最近触发会话”范围内追加约束，避免污染其他插件或对话。
        """
        umo = event.unified_msg_origin
        now = time.time()

        scope_until = float(self._llm_scope_until.get(umo, 0))
        if scope_until > now:
            return True

        # 兜底：若当前消息明确命中文件发送意图，也允许注入
        text = (event.message_str or "").strip()
        if not text:
            return False

        keywords = self._cfg(
            "llm_tool_config.intent_keywords",
            ["发我", "发一下", "传我", "发送文件", "给我文档", "把资料发来"],
        )
        if not isinstance(keywords, list):
            return False
        return any(str(k) in text for k in keywords)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        self._ensure_cleanup_task()
        self._prune_llm_scope()

        try:
            if not self._should_inject_prompt(event):
                return

            tool_desc = str(
                self._cfg(
                    "llm_tool_config.tool_description",
                    "在白名单目录内安全检索并发送文件；禁止越权路径访问。",
                )
            )
            intent_keywords = self._cfg("llm_tool_config.intent_keywords", [])
            kw_text = "、".join([str(x) for x in intent_keywords]) if isinstance(intent_keywords, list) else ""

            security_prompt = (
                "\n[文件发送插件约束]\n"
                "1) 仅可调用 send_local_file 工具处理“发送文件”意图。\n"
                "2) 严格禁止构造、建议或发起任何网络请求。\n"
                "3) 严格禁止危险执行与破坏行为（如 eval/exec、删除/篡改文件）。\n"
                "4) 严格禁止越权访问白名单外路径。\n"
                "5) 严格禁止输出政治、暴力、色情、违法（盗版/诈骗/赌博）内容，禁止恶意代码与隐私侵犯行为。\n"
                f"6) 工具描述：{tool_desc}\n"
                f"7) 意图关键词参考：{kw_text}\n"
            )

            req.system_prompt = (req.system_prompt or "") + security_prompt
        except Exception as e:
            self._log("warning", "on_llm_request 钩子处理失败: %s", str(e))

    async def terminate(self):
        self.pending_confirmations.clear()
        self._llm_scope_until.clear()
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass
        self._log("info", "插件已终止，状态已清理。")