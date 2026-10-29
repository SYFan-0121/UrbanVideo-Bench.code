"""仅解析模型回答中的 A–G 选项，不导入 Agentic/STRIVE。"""

import re


def extract_option(text: str) -> str | None:
    """
    从模型输出中提取选项字母（A-G）。
    支持格式：'Option: [A]' / 'Option: A' / 单独字母行
    """
    # 优先匹配标准格式
    m = re.search(r'Option\s*:\s*\[?\s*([A-Ga-g])\s*\]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 退化匹配：独立字母
    m = re.search(r'\b([A-Ga-g])\b', text)
    if m:
        return m.group(1).upper()
    return None
