# v0.3.8 指令参数截断修复验证

验证日期：2026-11-09。

## 故障与原因

用户输入 `/拼图 233 123` 时，插件持续返回用法。此前的业务测试直接把完整参数传给处理函数，没有经过 AstrBot 的指令解析。

AstrBot `CommandFilter.init_handler_md()` 在参数存在 Python 默认值时保存默认值，否则保存类型注解；随后只在参数标记是 `GreedyStr` 类本身时接收剩余全部文本。因此 `query: GreedyStr = ""` 会被当成普通字符串参数，只收到 `233`。

- 官方源码：[AstrBot command.py](https://github.com/AstrBotDevs/AstrBot/blob/master/astrbot/core/star/filter/command.py)。
- 本次回放的源码 Git blob：`abf339fda7f0740f542a91cd4adb859d5ef6ed5f`，已核对与验证当天的官方默认分支一致。
- 修改前，真实解析器回放得到 `{'query': '233'}`，回归断言预期 `{'query': '233 123'}`，确认失败原因。
- 修改后，移除 `/拼图`、`/随机拼图`、`/画廊`、`/生图`、`/生动图` 的 `GreedyStr` 默认值；框架仍会为无参数指令传入空字符串。

## 已验证场景

27 项路由测试在离线契约模式和真实 `CommandFilter` 源码模式下分别全部通过，覆盖：

- 两个、四个、二十个编号完整传入，制表符和换行按框架规则归一化。
- 从框架解析结果实际进入 `/拼图`，两图、三图和四图均生成临时 PNG，不重复入库。
- 画廊方括号中的多个编号、生图及动图中的多段提示词、命令别名保持完整。
- 无参数时保留空字符串，不触发框架的“必要参数缺失”错误。
- `/随机拼图 4 5` 保留完整输入后按插件规则拒绝，不静默变成 `/随机拼图 4`。
- 未唤醒事件和非本指令保持不匹配。

完整测试集使用真实解析器回放模式运行，435 项全部通过；Ruff、compileall、配置 JSON 校验与 `git diff --check` 同样通过。

## 重现方式

默认运行不依赖 AstrBot 安装或网络的指令契约测试：

```powershell
python -m pytest -q tests/test_command_routing.py
```

如有 AstrBot 源码，可使用其未修改的真实解析器回放同一组测试：

```powershell
$env:ASTRBOT_COMMAND_FILTER_SOURCE = 'C:\path\to\AstrBot\astrbot\core\star\filter\command.py'
python -m pytest -q tests/test_command_routing.py
```

真实模式只替换框架依赖的轻量对象，直接加载该源码文件中的 `CommandFilter` 和 `GreedyStr`。测试从框架已移除唤醒前缀的消息文本开始，覆盖参数解析、插件分派、实际图库读取及 Pillow 拼图；不代表完整 AstrBot/QQ 发送端到端测试，也不调用付费生图 API。
