latin = [
    ' ', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '[', ']', '{', '|', '}', '~',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]

science = [
    '§', '±', '°', '÷', 'Σ', 'Δ', 'ф', 'Ω', 'α', 'β', 'γ', 'η', 'θ', 'λ', 'μ',
    'σ', 'Ц', '→', '↓', '∞', '≒',
]

to_be_converted = [
    # µ is not μ
    # ‐ is not -
    '´', '²', '³', 'µ', '×', 'é', 'ö', 'Φ', 'ü', 'Γ', 'φ', '‐', '‘', '’', '“',
    '”','‥', '…', '⁰', '⁺', '⁻', '₀', '₁', '₂', '₃', '₄', '₅', '₇', '℀', '℃',
    '℉', 'ℓ', '№', 'Å', 'Å', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ',
    'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ', 'ⅴ', 'ⅵ', 'ⅶ', '−', '⊿', '△', '▵', '▶', '⑵', 'Ⓢ',
    '○', '◦', '｡', '。', '、', '，', '　', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧',
    '⑨', '⑩', '⑪', '⑫', '〃', '〇', '〈', '〉', '〔', '〕', '〜', '゜', '・', 'ー',
    '㎎', '㎏', '㎜', '㎝', '㎠', '㎡', '㎥', '㏄', '％', '（', '）', '＊', '※',
    '＋', '－', '．', '／', '０', '１', '２', '３', '４', '５', '６', '７', '８',
    '９', '：', '；', '＜', '＝', '＞', '？', '@', '＠', 'Ａ', 'Ｂ', 'Ｃ', 'Ｇ',
    'Ｈ', 'Ｋ', 'Ｏ', 'Ｐ', 'Ｓ', 'Ｗ', 'Ｘ', '［', '］', 'ｂ', 'ｃ', 'ｄ', 'ｇ',
    'ｈ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｒ', 'ｔ', 'ｖ', 'ｘ', '｛', '｝', '～', '､',
    '･', 'ｰ',
]

unknown = '⁇'
to_unknown = [
    '¥', '￥', '∴', '≡', '←', '↑', '↗', '↘', '↙', '⇄', '⇒', '⇓', '∀', '□',
    '◎', '「', '」', 'に', 'を', '値', '分', '図', '析', '残', '渣', '表',
]

convert_table = {
    '´': '\'', '²': '2', '³': '3', 'µ': 'v', '×': 'x', 'é': 'e', 'ö': 'o',
    'Φ': 'ф', 'ü': 'u', 'Γ': 'γ', 'φ': 'ф', '‐': '-', '‘': '\'', '’': '\'',
    '“': '\"', '”': '\"', '‥': '..', '…': '...', '⁰': '°', '⁺': '+', '⁻': '-',
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₇': '7',
    '℀': 'a/c', '℃': '°C', '℉': '°F', 'ℓ': 'l', '№': 'No.', 'Å': 'A', 'Å': 'A',
    'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V', 'Ⅵ': 'VI',
    'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'ⅰ': 'i', 'ⅱ': 'ii', 'ⅲ': 'iii',
    'ⅳ': 'iv', 'ⅴ': 'v', 'ⅵ': 'vi', 'ⅶ': 'vii', '−': '-', 'Ⓢ': 'S', '①': '1',
    '②': '2', '③': '3', '④': '4', '⑤': '5', '⑥': '6', '⑦': '7', '⑧': '8',
    '⑨': '9', '⑩': '10', '⑪': '11', '⑫': '12', '⑵': '[2]', '⊿': 'Δ', '△': 'Δ',
    '▵': 'Δ', '▶': 'Δ' , '○': 'o', '◦': 'o', '｡': '.', '。': '.', '　': ' ',
    '、': ',', '〃': '\"', '〇': 'O', '〈': '[', '〉': ']', '〔': '[', '〕': ']',
    '〜': '~', '゜': '°', '・': '.', 'ー': '-', '㎎': 'mg', '㎏': 'kg', '㎜': 'mm',
    '㎝': 'cm', '㎠': 'cm2', '㎡': 'm2', '㎥': 'm3', '㏄': 'cc', '％': '%',
    '（': '(', '）': ')', '＊': '*', '※': '*', '＋': '+', '，': '.', '－': '-',
    '．': '.', '／': '/', '０': '0', '１': '1', '２': '2', '３':'3', '４':'4',
    '５': '5', '６':'6', '７':'7', '８':'8', '９':'9', '：': ':', '；': ';',
    '＜': '<', '＝': '=', '＞': '>', '？': '?', '@': 'a', '＠': 'a', 'Ａ': 'A',
    'Ｂ': 'B', 'Ｃ': 'C', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｋ': 'K', 'Ｏ': 'O', 'Ｐ': 'P',
    'Ｓ': 'S', 'Ｗ': 'W', 'Ｘ': 'X', '［': '[', '］': ']', 'ｂ': 'b', 'ｃ': 'c',
    'ｄ': 'd', 'ｇ': 'g', 'ｈ': 'h', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
    'ｒ': 'r', 'ｔ': 't', 'ｖ': 'v', 'ｘ': 'x', '｛': '{', '｝': '}', '～': '~',
    '､': '.', '･': '.', 'ｰ': '-'
}

alphabet = latin + science + [unknown]

for c in to_unknown:
    convert_table[c] = unknown
for c in alphabet:
    convert_table[c] = c

# alphabet = [
#     " ", "\"", "%", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", "=", "~",
#     "°", "→",
#     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
#     "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
#     "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
#     "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
#     unknown
# ]

alphabet_index = {alphabet[i]: i for i in range(len(alphabet))}
blank_index = len(alphabet)