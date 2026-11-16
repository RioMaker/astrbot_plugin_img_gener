from __future__ import annotations

from dataclasses import dataclass, field

PING_SCOPE = (
    "仅检测生图 API 的模型列表接口，不生成图片、不占生图额度；"
    "实际出图、参考图上传和安全审核是否成功仍需实际请求验证。"
)


@dataclass(frozen=True, slots=True)
class ImagePingResult:
    status: str
    message: str
    elapsed_ms: int | None = None
    http_status: int | None = None
    models: dict[str, bool] = field(default_factory=dict)

    def report(self) -> str:
        label = {"ok": "通过", "warning": "待确认", "error": "失败"}[self.status]
        lines = [f"生图 ping：{label}", self.message]
        details = []
        if self.elapsed_ms is not None:
            details.append(f"耗时：{self.elapsed_ms} ms")
        if self.http_status is not None:
            details.append(f"HTTP {self.http_status}")
        if details:
            lines.append(" · ".join(details))
        for model, visible in self.models.items():
            lines.append(f"模型 {model}：{'已列出' if visible else '未列出（需确认分组/权限）'}")
        lines.append(PING_SCOPE)
        return "\n".join(lines)
