from __future__ import annotations


def mask_phone(phone: str) -> str:
    if len(phone) > 4:
        return phone[:-4] + "****"
    return "****"
