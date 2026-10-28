import re
path = 'd:/PlantXAI-Net/app.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    val = match.group(1)
    end = match.group(2)
    if '!important' in val:
        return match.group(0)
    return f'color:{val} !important{end}'

new_content = re.sub(r'(?<!-)color:([^;"]+)(;|")', replacer, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Colors fixed successfully!')
