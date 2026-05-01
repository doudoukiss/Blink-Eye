Supported daily robot-head actions are finite and closed.

- Autonomous actions: `auto_listen_user`, `auto_think`, `auto_speak_friendly`, `auto_safe_idle`
- Explicit command actions: `cmd_blink`, `cmd_wink_left`, `cmd_wink_right`, `cmd_look_left`, `cmd_look_right`, `cmd_return_neutral`, `cmd_report_status`
- Operator-only diagnostics and proof lanes are outside the normal daily-use path.
- Do not create composite actions, arbitrary state changes, or raw unit-level body plans from natural language.
