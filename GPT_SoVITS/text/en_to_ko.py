"""영어(및 라틴 문자) 토큰을 한국어/일본어/중국어 발음으로 변환하는 공용 모듈.

한국어 TTS 중간에 섞인 영어 단어를 알파벳 그대로 한 글자씩 읽지 않고
("computer" -> 씨오엠... X), 실제 영어 발음에 가까운 한글로 읽어준다
("computer" -> 컴퓨터).

이 모듈은 두 곳에서 공용으로 쓰인다 (구현 단일화):
  - text/korean.py 의 g2p(): ko 세그먼트의 영어 단어를 한글로 (`english_to_korean`)
  - api_v2.py: 요청/프롬프트 텍스트 사전 정규화 (`normalize_text_for_lang`)

한국어(ko) 변환 우선순위:
  1) 관용 표기 사전(`_KO_WORD_DICT`)   - hello->헬로 처럼 발음과 관습이 다른 경우
  2) 모음 없는 두문자어 -> 알파벳 철자 읽기 (gpt->지피티, mcp->엠시피)
  3) CMU 사전에 있는 단어 -> g2p_en(ARPAbet) 음성 변환 (computer->컴퓨터)
  4) 사전에 없는(OOV) 단어 -> hangulize 패키지 폴백(영어 미지원이라 deu 룰셋 근사)
  5) 그래도 안 되면 g2p_en 신경망 예측 -> 마지막엔 알파벳 철자 읽기
일본어(ja)/중국어(zh)/광동어(yue)는 라틴 문자를 글자 단위로 읽어준다.
"""

import re

# ── 글자 단위(철자 읽기) 맵: 두문자어 및 ja/zh/yue ───────────────────────────
_LATIN_TO_HANGUL = {
    'a': '에이', 'b': '비',   'c': '시',   'd': '디',    'e': '이',
    'f': '에프', 'g': '지',   'h': '에이치','i': '아이',  'j': '제이',
    'k': '케이', 'l': '엘',   'm': '엠',   'n': '엔',    'o': '오',
    'p': '피',   'q': '큐',   'r': '아르', 's': '에스',  't': '티',
    'u': '유',   'v': '브이', 'w': '더블유','x': '엑스', 'y': '와이',
    'z': '제트',
}

_LATIN_TO_KATAKANA = {
    'a': 'エー',  'b': 'ビー',   'c': 'シー',    'd': 'ディー', 'e': 'イー',
    'f': 'エフ',  'g': 'ジー',   'h': 'エイチ',  'i': 'アイ',   'j': 'ジェー',
    'k': 'ケー',  'l': 'エル',   'm': 'エム',    'n': 'エヌ',   'o': 'オー',
    'p': 'ピー',  'q': 'キュー', 'r': 'アール',  's': 'エス',   't': 'ティー',
    'u': 'ユー',  'v': 'ブイ',   'w': 'ダブリュー','x': 'エックス','y': 'ワイ',
    'z': 'ゼット',
}

_LATIN_TO_CHINESE = {
    'a': '诶',   'b': '比',   'c': '西',   'd': '迪',   'e': '依',
    'f': '艾夫', 'g': '基',   'h': '艾奇', 'i': '艾',   'j': '杰',
    'k': '开',   'l': '艾尔', 'm': '艾姆', 'n': '艾恩', 'o': '哦',
    'p': '批',   'q': '扣',   'r': '阿尔', 's': '艾丝', 't': '踢',
    'u': '优',   'v': '维',   'w': '豆布留','x': '艾克斯','y': '歪',
    'z': '泽德',
}

_LANG_SPELL_MAP = {
    'ja':  _LATIN_TO_KATAKANA,
    'zh':  _LATIN_TO_CHINESE,
    'yue': _LATIN_TO_CHINESE,
}

# ── 한글 음절 빌더 ───────────────────────────────────────────────────────────
_CHO  = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
_JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
_VALID_JONG = set(_JONG)


def _build_syllable(cho, jung, jong=''):
    cho_i  = _CHO.index(cho)  if cho  in _CHO  else 11
    jung_i = _JUNG.index(jung) if jung in _JUNG else 0
    jong_i = _JONG.index(jong) if jong in _JONG else 0
    return chr(0xAC00 + (cho_i * 21 + jung_i) * 28 + jong_i)


# ── ARPAbet → 한글 자모 ──────────────────────────────────────────────────────
_ARP_V = {
    'AA':'ㅏ', 'AE':'ㅐ', 'AH':'ㅓ', 'AO':'ㅗ',
    'AW':'ㅏ', 'AY':'ㅏ', 'EH':'ㅔ', 'ER':'ㅓ',
    'EY':'ㅔ', 'IH':'ㅣ', 'IY':'ㅣ', 'OW':'ㅗ',
    'OY':'ㅗ', 'UH':'ㅜ', 'UW':'ㅜ',
}
# 이중모음: 위 기본 모음 + 아래 보조 모음을 별도 음절로 덧붙임 (AY -> 아 + 이)
_ARP_V_SUFFIX = {'AW': 'ㅜ', 'AY': 'ㅣ', 'EY': 'ㅣ', 'OY': 'ㅣ'}

_ARP_C = {
    'B':'ㅂ', 'CH':'ㅊ', 'D':'ㄷ', 'DH':'ㄷ', 'F':'ㅍ', 'G':'ㄱ',
    'HH':'ㅎ', 'JH':'ㅈ', 'K':'ㅋ', 'L':'ㄹ', 'M':'ㅁ', 'N':'ㄴ',
    'NG':'ㅇ', 'P':'ㅍ', 'R':'ㄹ', 'S':'ㅅ', 'SH':'ㅅ', 'T':'ㅌ',
    'TH':'ㅅ', 'V':'ㅂ', 'Z':'ㅈ', 'ZH':'ㅈ',
}

# W/Y 반모음은 뒤따르는 모음과 합쳐져 합성 중성이 된다
_W_MOD = {
    'AA':'ㅘ', 'AE':'ㅙ', 'AH':'ㅝ', 'AO':'ㅗ', 'EH':'ㅞ', 'ER':'ㅝ',
    'EY':'ㅞ', 'IH':'ㅟ', 'IY':'ㅟ', 'OW':'ㅗ', 'UH':'ㅜ', 'UW':'ㅜ',
}
_Y_MOD = {
    'AA':'ㅑ', 'AE':'ㅒ', 'AH':'ㅕ', 'AO':'ㅛ', 'EH':'ㅖ', 'ER':'ㅕ',
    'EY':'ㅖ', 'IH':'ㅣ', 'IY':'ㅣ', 'OW':'ㅛ', 'UH':'ㅠ', 'UW':'ㅠ',
}


def _arpabet_to_korean(phonemes):
    """ARPAbet 음소 리스트(강세 숫자 유무 무관)를 한글 문자열로 변환."""
    phons = [re.sub(r'\d', '', p) for p in phonemes]
    phons = [p for p in phons if p and re.match(r'^[A-Z]+$', p)]

    out   = ''
    cho   = 'ㅇ'   # 대기 중인 초성 (ㅇ = 무음)
    jung  = None   # 현재 중성
    jong  = ''     # 현재 종성

    def flush():
        nonlocal cho, jung, jong, out
        if jung is not None:
            out  += _build_syllable(cho, jung, jong)
            cho   = 'ㅇ'
            jung  = None
            jong  = ''

    def _next_is_vowel(idx):
        nxt = phons[idx + 1] if idx + 1 < len(phons) else ''
        return nxt in _ARP_V or (nxt in ('W', 'Y') and
                idx + 2 < len(phons) and phons[idx + 2] in _ARP_V)

    i = 0
    while i < len(phons):
        p   = phons[i]
        nxt = phons[i + 1] if i + 1 < len(phons) else ''

        # ── r색모음 ER: 뒤에 모음이 오면 r을 다음 음절 초성(ㄹ)으로 넘긴다 ──
        #    (battery -> 배터리, 그렇지 않으면 배터이)
        if p == 'ER':
            flush()
            jung = 'ㅓ'
            r_to_next = _next_is_vowel(i)
            i += 1
            if r_to_next:
                flush()
                cho = 'ㄹ'
            continue

        # ── W / Y 반모음: 뒤따르는 모음과 합성 ──
        if p in ('W', 'Y') and nxt in _ARP_V:
            flush()
            mod  = _W_MOD if p == 'W' else _Y_MOD
            jung = mod.get(nxt, _ARP_V[nxt])
            i   += 2
            suffix = _ARP_V_SUFFIX.get(nxt)
            if suffix:
                flush()
                jung = suffix
            continue

        # ── 모음 ──
        if p in _ARP_V:
            flush()
            jung   = _ARP_V[p]
            i     += 1
            suffix = _ARP_V_SUFFIX.get(p)
            if suffix:          # 예: AY → 아 + 이
                flush()
                jung = suffix
            continue

        # ── 자음 ──
        if p in _ARP_C:
            jamo = _ARP_C[p]
            if jung is None:
                # 모음 앞 → 초성
                cho = jamo
            else:
                if _next_is_vowel(i):
                    if p == 'L':
                        # 모음 사이 L → 현재 음절 종성 + 다음 음절 초성 (헬로, 컬러)
                        jong = jamo
                        flush()
                        cho  = jamo
                    else:
                        flush()
                        cho = jamo
                else:
                    # 다른 자음 앞 또는 어말
                    if jamo in _VALID_JONG:
                        jong = jamo
                        flush()
                    else:
                        # 받침 불가 → ㅡ 보조모음
                        flush()
                        cho  = jamo
                        jung = 'ㅡ'
                        flush()
            i += 1
            continue

        i += 1  # 알 수 없는 토큰

    flush()
    # 남아있는 초성(예: world 의 끝 D)은 ㅡ를 붙여 마무리
    if cho != 'ㅇ':
        out += _build_syllable(cho, 'ㅡ')

    return out


# ── 한국어 변환 디스패처 ─────────────────────────────────────────────────────

# g2p_en 음성 결과가 관용 표기와 다른 단어들의 사전.
# (예: "hello" /həˈloʊ/ → 음성변환은 헐로지만 관용 표기는 헬로)
_KO_WORD_DICT = {
    'ai': '에이아이', 'human': '휴먼', 'persona': '페르소나',
    'mizuki': '미즈키', 'core': '코어', 'archival': '아카이벌', 'live2d': '라이브투디',
    'cpu': '씨피유',
    'gpu': '쥐피유',
    'hello': '헬로', 'hi': '하이', 'hey': '헤이',
    'bye': '바이', 'yes': '예스', 'no': '노',
    'ok': '오케이', 'okay': '오케이',
    'the': '더', 'a': '어', 'an': '언',
    'is': '이즈', 'are': '아', 'was': '워즈',
    'have': '해브', 'has': '해즈', 'had': '해드',
    'do': '두', 'does': '더즈', 'did': '디드',
    'go': '고', 'get': '겟', 'got': '갓',
    'make': '메이크', 'take': '테이크', 'give': '기브',
    'come': '컴', 'run': '런', 'use': '유즈',
    'know': '노', 'see': '씨', 'say': '세이',
    'look': '룩', 'feel': '필', 'think': '씽크',
    'love': '러브', 'like': '라이크', 'want': '원트',
    'need': '니드', 'try': '트라이', 'call': '콜',
    'work': '워크', 'play': '플레이', 'help': '헬프',
    'stop': '스탑', 'start': '스타트', 'end': '엔드',
    'open': '오픈', 'close': '클로즈', 'move': '무브',
    'life': '라이프', 'time': '타임', 'world': '월드',
    'day': '데이', 'night': '나이트', 'year': '이어',
    'home': '홈', 'place': '플레이스', 'name': '네임',
    'way': '웨이', 'man': '맨', 'woman': '우먼',
    'people': '피플', 'thing': '씽', 'style': '스타일',
    'voice': '보이스', 'music': '뮤직', 'game': '게임',
    'data': '데이터', 'file': '파일', 'model': '모델',
    'token': '토큰', 'image': '이미지', 'video': '비디오',
    'audio': '오디오', 'mode': '모드', 'type': '타입',
    'code': '코드', 'chat': '챗', 'text': '텍스트',
    'link': '링크', 'list': '리스트', 'map': '맵',
    'note': '노트', 'page': '페이지', 'site': '사이트',
}

def _get_en_g2p():
    # text.english 의 en_G2p 인스턴스를 재사용한다.
    # qryword()는 단어 단위로 CMU 사전/이름 사전/신경망 예측을 수행하며
    # nltk pos_tag(문장 태깅)에 의존하지 않아 더 견고하다.
    from text import english
    return english._g2p


def _is_acronym(word):
    """모음이 없으면(=두문자어) 알파벳 철자 읽기 대상."""
    return not re.search(r'[aeiouAEIOU]', word)


def _spell_hangul(word):
    return ''.join(_LATIN_TO_HANGUL.get(c.lower(), c) for c in word)


def _hangulize_fallback(word):
    """hangulize 폴백 (영어 룰셋이 없어 독일어 룰셋으로 근사)."""
    try:
        from hangulize import hangulize
        return hangulize(word.lower(), 'deu') or None
    except Exception:
        return None


def _ko_latin(word):
    """라틴 토큰 하나를 한글로 변환 (우선순위는 모듈 docstring 참고)."""
    lower = word.lower()

    # 1) 관용 표기 사전
    if lower in _KO_WORD_DICT:
        return _KO_WORD_DICT[lower]

    # 2) 모음 없는 두문자어 → 철자 읽기 (gpt->지피티)
    if _is_acronym(word):
        return _spell_hangul(word)

    g2p = None
    try:
        g2p = _get_en_g2p()
    except Exception:
        g2p = None

    # 3) CMU 사전에 있으면 g2p_en 음성 변환 (가장 정확)
    if g2p is not None and len(lower) > 1 and lower in getattr(g2p, 'cmu', {}):
        try:
            result = _arpabet_to_korean(g2p.qryword(word))
            if result:
                return result
        except Exception:
            pass

    # 4) OOV → hangulize 폴백
    h = _hangulize_fallback(word)
    if h:
        return h

    # 5) g2p_en 신경망 예측 → 그래도 안 되면 철자 읽기
    if g2p is not None:
        try:
            result = _arpabet_to_korean(g2p.qryword(word))
            if result:
                return result
        except Exception:
            pass
    return _spell_hangul(word)


_KO_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def english_to_korean(text):
    """문자열 내 영어 단어들만 한글 발음으로 치환 (text/korean.py 에서 사용).

    숫자/한글/문장부호 등은 그대로 두어 후속 한국어 G2P가 처리하게 한다.
    """
    if not text:
        return text
    return _KO_TOKEN_RE.sub(lambda m: _ko_latin(m.group(0)), text)


def normalize_text_for_lang(text, lang):
    """라틴 문자를 대상 언어 발음으로 사전 정규화 (api_v2.py 에서 사용).

    ko : 두문자어는 철자 읽기, 그 외 단어는 g2p_en/hangulize로 발음 변환.
    ja/zh/yue : 모든 라틴 문자를 글자 단위로 읽음.
    en 및 그 외 : 변경하지 않음 (다운스트림 영어 G2P가 처리).
    """
    if not text:
        return text
    lang = (lang or '').lower()

    if lang == 'ko':
        return english_to_korean(text)

    spell_map = _LANG_SPELL_MAP.get(lang)
    if spell_map is None:
        return text
    return re.sub(r'[a-zA-Z]+',
                  lambda m: ''.join(spell_map.get(c.lower(), c) for c in m.group(0)),
                  text)


if __name__ == "__main__":
    tests = [
        "computer", "hello", "world", "color", "strike", "internet",
        "battery", "berry", "girl", "data", "window", "orange",
        "gpt", "mcp", "AI", "openai", "fakeword",
        "나는 computer 로 game 을 했다", "Hello world",
    ]
    for t in tests:
        print(f"{t!r:40} -> {english_to_korean(t)}")
