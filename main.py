a = "a"
b= "b"
c= "c"

tools = [a, b, c]


tools_by_name = {t.name: t for t in tools}
print(tools_by_name)