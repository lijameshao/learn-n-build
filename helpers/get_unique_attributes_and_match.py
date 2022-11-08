"""
Goals:
    - To find set of unique attribute combinations
    - Find new entries that have the same sub-set of attributes as existing entry
    - without using Pandas
"""


from pydantic import BaseModel


class Foo(BaseModel):
    str_attr: str
    int_attr: int
    lst_attr: list
    oth_attr: str


class Bar(BaseModel):
    attr_str: str
    attr_int: int
    attr_lst: list
    attr_oth: str


# Existing entries
entry_a = Foo(str_attr="a", int_attr=1, lst_attr=["label_a"], oth_attr="o")
entry_b = Foo(str_attr="a", int_attr=1, lst_attr=["label_a"], oth_attr="o")
entry_c = Foo(str_attr="b", int_attr=1, lst_attr=["label_a"], oth_attr="o")

# New entries to test against
a_entry = Bar(attr_str="a", attr_int=1, attr_lst=["label_a"], attr_oth="p")
b_entry = Bar(attr_str="c", attr_int=1, attr_lst=["label_a"], attr_oth="p")


existing_entries = [entry_a, entry_b, entry_c]
new_entries = [a_entry, b_entry]


unique_combinations = []
for entry in existing_entries:
    entry_dict = entry.dict(include={"str_attr", "int_attr", "lst_attr"})
    if entry_dict in unique_combinations:
        continue
    unique_combinations.append(entry_dict)


def map_bar_to_foo(b: Bar) -> Foo:
    return Foo(
        str_attr=b.attr_str,
        int_attr=b.attr_int,
        lst_attr=b.attr_lst,
        oth_attr="does not matter",
    )


matched = []
for entry in new_entries:
    mapped_entry = map_bar_to_foo(entry)
    mapped_entry_dict = mapped_entry.dict(include={"str_attr", "int_attr", "lst_attr"})
    if mapped_entry_dict in unique_combinations:
        matched.append(entry)
