"""Short descriptions of prompt templates for the Prompts panel (ARCHIE-208).

Kept here, not in the templates, so a description edit does not change the
template hash or the rendered prompt. Key: template file stem.
"""

PROMPT_DESCRIPTIONS: dict[str, str] = {
    "response_format_router": "Auto: JEV выбирает формат после оркестрации и сбора данных",
    "response_format_plain": "Auto: критерии выбора простого текста",
    "response_format_level2_answer": "Auto: критерии выбора форматированного текста",
    "response_format_ui_answer": "Auto: критерии выбора UI-ответа",
    "cmd_prompt": "Stage 1: решает, ответить сразу, вызвать tools или уточнить; выбирает tools и UI-интенты",
    "assistant_prompt": "Общий контекст Stage 3: профиль пользователя, персона, город, язык, форматы",
    "format_ui_answer": "Stage 3, ui_answer: генеративный UI — карточки, формы, таблицы, быстрые действия",
    "format_level2_answer": "Stage 3, level2_answer: текст в markdown и кнопки быстрых действий",
    "format_level3_answer": "Stage 3, level3_answer: текст, один встроенный виджет и кнопки",
    "format_plain": "Stage 3, plain и voice: простой текст без разметки",
    "format_formatted_text": "Stage 3, markdown/html/json и др.: текст в заданном формате",
    "format_gemini_tts": "Stage 3, gemini_tts: сценарий для озвучки голосом",
    "format_dashboard": "Без Stage 1: плитки дашборда умного дома",
    "format_widget": "Без Stage 1: одиночный виджет без текста",
    "persona_business": "Персона «Бизнес»: коротко, по делу, профессионально",
    "persona_butler": "Персона «Дворецкий»: вежливый викторианский слуга",
    "persona_bro": "Персона «Бро»: расслабленный дружелюбный городской стиль",
    "persona_flirty": "Персона «Флирт»: игривый тёплый тон",
    "persona_futurebot": "Персона «Робот из будущего»: точный, слегка механический",
    "swarm_base": "Рой: общая инструкция для любого агента роя",
    "agent_light": "Рой, @light: управление светом",
    "agent_climate": "Рой, @climate: управление климатом",
    "agent_search": "Рой, @search: веб-поиск — погода, места, новости, факты",
    "agent_voice": "Рой, @voice: озвучивает алерты и просьбы на колонке",
    "agent_display": "Рой, @display: показывает алерты и просьбы на LED-табло",
    "watch_task": "Рой: спонтанная проверка (дом, почта, календарь, футбол) и важность алерта",
    "group_council": "Рой, совет: роль участника совета (@bro, @business, @flirty)",
    "council_triage": "Совет: председатель решает, созывать ли совет или ответить сам",
    "council_opinion": "Совет: позиция участника в обсуждении",
    "council_decision": "Совет: итоговый ответ пользователю и поручения агентам",
    "council_rules": "Совет: правила итогового решения и поручений",
    "council_alert": "Совет: решает, действовать ли дому самому по оповещению проверки",
}
