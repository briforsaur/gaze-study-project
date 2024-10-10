

def fix_datetime_string(dt_str: str) -> str:
    """Fixes datetime strings with a typo to match ISO format
    
    The datetime strings recorded in the experiment had a typo in the separator between
    the date and the time such that they do not match ISO format.
    Correct ISO format: YYYY-MM-DDTHH:MM:SS
    Typo:               YYYY-MM-DD-THH:MM:SS

    This function fixes the typo and returns a properly formatted ISO datetime string.
    """
    dt_str_parts = dt_str.split("-")
    return "-".join(dt_str_parts[0:-1]) + dt_str_parts[-1]


def make_digit_str(num: int, width: int = 3) -> str:
    """Convert an integer to a fixed-width string padded with zeros

    Examples: 
    >>> make_digit_str(1)
    '001'
    >>> make_digit_str(1, 4)
    '0001'
    >>> make_digit_str(12, 4)
    '0012'
    """
    return "{:0={width}}".format(num, width=width)


if __name__=="__main__":
    typo_str = "2024-09-01-T12:00:00"
    print(f"Typo string: {typo_str}")
    print(f"Corrected:   {fix_datetime_string(typo_str)}")
    print(make_digit_str(1))
    print(make_digit_str(10))
    print(make_digit_str(100))
    print(make_digit_str(100, 5))