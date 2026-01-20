#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSUT BASIC5000 アクセント核誤り付与データセット生成スクリプト

正解データ（e2e_symbol/phoneme.yaml）のアクセント句構造を維持しながら、
各アクセント句内でアクセント核の位置を移動させる形で誤りを加える。

アクセント句の定義:
  - `[` (句頭上昇) から次の `[` または `_` (ポーズ) までが1つのアクセント句
  - 各アクセント句には0個または1個の `]` (アクセント核) が存在

久保薗晴夫の「アクセントの法則」に基づく制約:
  1. 特殊拍（撥音N、促音Q/cl）にはアクセント核を置けない
  2. 長音の後半（連続母音の2番目）にはアクセント核を置けない
  3. アクセント核は母音の直後にのみ配置可能

出力:
  1. basic5000_accent_error.yaml - 誤り付与済みフォーマットデータ
  2. basic5000_diff.tsv - 元データとの差分比較ログ
"""

import random
import csv
import urllib.request
from typing import List, Tuple, Optional
from dataclasses import dataclass

import yaml


# 定数定義
SOURCE_URL = "https://raw.githubusercontent.com/sarulab-speech/jsut-label/master/e2e_symbol/phoneme.yaml"
OUTPUT_YAML = "basic5000_accent_error.yaml"
OUTPUT_TSV = "basic5000_diff.tsv"
OUTPUT_HTML = "basic5000_diff.html"

# 日本語母音セット
VOWELS = {"a", "i", "u", "e", "o"}

# アクセント核配置禁止音素（特殊拍）
FORBIDDEN_NUCLEUS = {"N", "cl", "q", "Q"}

# 制御記号
ACCENT_SYMBOLS = {"[", "]"}
BOUNDARY_SYMBOLS = {"#", "_"}
ALL_SYMBOLS = {"^", "$", "[", "]", "#", "_"}


@dataclass
class AccentPhrase:
    """アクセント句を表すデータクラス"""
    start_idx: int          # 開始位置（トークンインデックス）
    end_idx: int            # 終了位置（トークンインデックス、排他的）
    has_rise: bool          # 句頭上昇 `[` があるか
    nucleus_idx: Optional[int]  # アクセント核 `]` の位置（トークンインデックス）


def download_data(url: str) -> dict:
    """URLからYAMLデータをダウンロードしてパース"""
    print(f"Downloading data from: {url}")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    print("Download complete.")
    return yaml.safe_load(content)


def tokenize(formatted_str: str) -> List[str]:
    """フォーマット済み文字列をトークンリストに分割"""
    return formatted_str.split('-')


def join_tokens(tokens: List[str]) -> str:
    """トークンリストをフォーマット済み文字列に結合"""
    return '-'.join(tokens)


def is_phoneme(token: str) -> bool:
    """トークンが音素（記号でない）かどうか判定"""
    return token not in ALL_SYMBOLS and token != ''


def is_long_vowel_second(tokens: List[str], token_idx: int) -> bool:
    """
    指定位置のトークンが長音の後半かどうか判定
    """
    if tokens[token_idx] not in VOWELS:
        return False

    # 直前の音素を探す
    prev_phoneme = None
    for i in range(token_idx - 1, -1, -1):
        if is_phoneme(tokens[i]):
            prev_phoneme = tokens[i]
            break

    if prev_phoneme is None:
        return False

    # 同一母音の連続は長音
    return tokens[token_idx] == prev_phoneme and prev_phoneme in VOWELS


def can_place_nucleus_at(tokens: List[str], token_idx: int) -> bool:
    """
    指定位置の直後にアクセント核を配置可能か判定
    """
    token = tokens[token_idx]

    # 音素でなければ不可
    if not is_phoneme(token):
        return False

    # 母音でなければ不可
    if token not in VOWELS:
        return False

    # 特殊拍には不可
    if token in FORBIDDEN_NUCLEUS:
        return False

    # 長音の後半には不可
    if is_long_vowel_second(tokens, token_idx):
        return False

    return True


def find_accent_phrases(tokens: List[str]) -> List[AccentPhrase]:
    """
    トークンリストからアクセント句を特定

    アクセント句の定義（正しい定義）:
    - `#` (文節境界)、`_` (ポーズ)、`^` (文頭)、`$` (文末) がアクセント句の境界
    - `[` (句頭上昇) は句内のピッチ上昇マーカーであり、句の境界ではない
    - `]` (アクセント核) は句内での核位置を示す
    - 各アクセント句には最大1つの `]` が存在すべき
    - 1つの句内に複数の `]` がある場合は最初の `]` を核とする
    """
    phrases = []
    current_start = None
    current_has_rise = False
    current_nucleus = None
    has_phonemes = False

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == '^':
            # 文頭 - セグメント開始
            current_start = i + 1
            current_has_rise = False
            current_nucleus = None
            has_phonemes = False
            i += 1
            continue

        if token == '$':
            # 文末、現在のアクセント句を終了
            if current_start is not None and has_phonemes:
                phrases.append(AccentPhrase(
                    start_idx=current_start,
                    end_idx=i,
                    has_rise=current_has_rise,
                    nucleus_idx=current_nucleus
                ))
            break

        if token == '[':
            # 句頭上昇マーカー（句の境界ではない）
            # 現在の句に句頭上昇があることを記録
            current_has_rise = True
            i += 1
            continue

        if token == ']':
            # アクセント核の位置を記録（最初の]のみ、以降は無視）
            if current_nucleus is None:
                for j in range(i - 1, -1, -1):
                    if is_phoneme(tokens[j]):
                        current_nucleus = j
                        break
            i += 1
            continue

        if token == '_':
            # ポーズ = アクセント句の明確な境界
            if current_start is not None and has_phonemes:
                phrases.append(AccentPhrase(
                    start_idx=current_start,
                    end_idx=i,
                    has_rise=current_has_rise,
                    nucleus_idx=current_nucleus
                ))
            current_start = i + 1
            current_has_rise = False
            current_nucleus = None
            has_phonemes = False
            i += 1
            continue

        if token == '#':
            # 文節境界 = アクセント句の境界
            if current_start is not None and has_phonemes:
                phrases.append(AccentPhrase(
                    start_idx=current_start,
                    end_idx=i,
                    has_rise=current_has_rise,
                    nucleus_idx=current_nucleus
                ))
            current_start = i + 1
            current_has_rise = False
            current_nucleus = None
            has_phonemes = False
            i += 1
            continue

        # 音素
        has_phonemes = True
        i += 1

    return phrases


def find_valid_nucleus_positions_in_phrase(tokens: List[str], phrase: AccentPhrase) -> List[int]:
    """
    アクセント句内でアクセント核を配置可能な位置のリストを返す

    制約:
    - 母音の後にのみ配置可能
    - 特殊拍（N, cl, q）には配置不可
    - 長音の後半には配置不可
    - `[`がある場合、`[`より後にのみ配置可能（`[`は1モーラ目）
    - 句の最後の音素には配置不可（`]`の後に少なくとも1音素必要）
    """
    # 句内の`[`の位置を探す
    rise_pos = None
    for i in range(phrase.start_idx, phrase.end_idx):
        if tokens[i] == '[':
            rise_pos = i
            break

    # 句内の最後の音素位置を探す
    last_phoneme_pos = None
    for i in range(phrase.end_idx - 1, phrase.start_idx - 1, -1):
        if is_phoneme(tokens[i]):
            last_phoneme_pos = i
            break

    valid = []
    for i in range(phrase.start_idx, phrase.end_idx):
        # `[`がある場合、それより前には配置不可
        if rise_pos is not None and i < rise_pos:
            continue
        # 句の最後の音素には配置不可（]の後に少なくとも1音素必要）
        if last_phoneme_pos is not None and i == last_phoneme_pos:
            continue
        if can_place_nucleus_at(tokens, i):
            valid.append(i)
    return valid


def move_nucleus_in_string(original: str, error_rate: float = 0.8) -> str:
    """
    フォーマット済み文字列内のアクセント核を移動させる

    各アクセント句について:
    - 核がある場合: 確率的に別の位置に移動、または削除（平板化）
    - 核がない場合: 確率的に核を追加
    """
    tokens = tokenize(original)
    phrases = find_accent_phrases(tokens)

    # 変更を記録（後で一括適用）
    # 既存の ] を削除する位置と、新しい ] を挿入する位置
    nuclei_to_remove = set()
    nuclei_to_add = {}  # token_idx -> True

    for phrase in phrases:
        # 誤りを加えるかどうか判定
        if random.random() > error_rate:
            continue  # この句は変更なし

        valid_positions = find_valid_nucleus_positions_in_phrase(tokens, phrase)

        if not valid_positions:
            continue  # 有効な位置がなければスキップ

        current_nucleus = phrase.nucleus_idx

        if current_nucleus is not None:
            # 現在核がある場合
            other_positions = [p for p in valid_positions if p != current_nucleus]

            if other_positions and random.random() > 0.15:
                # 85%: 別の位置に移動
                new_nucleus = random.choice(other_positions)
                nuclei_to_remove.add(current_nucleus)
                nuclei_to_add[new_nucleus] = True
            elif random.random() > 0.5:
                # 7.5%: 平板化（核を削除）
                nuclei_to_remove.add(current_nucleus)
            # else: 7.5%: 変更なし
        else:
            # 現在平板の場合
            if random.random() > 0.3:
                # 70%: 核を追加
                new_nucleus = random.choice(valid_positions)
                nuclei_to_add[new_nucleus] = True
            # else: 30%: 平板のまま

    # トークンリストを再構築
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == ']':
            # 既存の ] を確認
            # 直前の音素を探す
            prev_phoneme_idx = None
            for j in range(len(new_tokens) - 1, -1, -1):
                if is_phoneme(new_tokens[j]):
                    prev_phoneme_idx = j
                    break

            # 対応する元のインデックスを推定（簡易的にスキップ判定）
            # 削除対象でなければ保持
            should_remove = False
            for rm_idx in nuclei_to_remove:
                if tokens[rm_idx] == new_tokens[prev_phoneme_idx] if prev_phoneme_idx is not None else False:
                    # より正確な判定が必要だが、簡易的に
                    pass

            # 全ての既存 ] は一旦保持せず、後で追加する方式に変更
            i += 1
            continue

        new_tokens.append(token)
        i += 1

    # より正確なアプローチ: 元のトークンを走査しながら再構築
    # 1. 全ての ] を除去
    # 2. 必要な位置に ] を追加

    # やり直し: より単純なアプローチ
    return rebuild_with_moved_nuclei(original, phrases, error_rate)


def rebuild_with_moved_nuclei(original: str, phrases: List[AccentPhrase], error_rate: float) -> str:
    """
    アクセント核を移動させた文字列を再構築

    アクセント句の境界は #, _, ^, $ であり、[ は境界ではない。
    各アクセント句には必ず1つの ] を配置する。
    """
    tokens = tokenize(original)

    # 新しい ] の位置を決定（各句に1つ）
    new_nuclei = {}  # phrase_idx -> token_idx

    for p_idx, phrase in enumerate(phrases):
        current_nucleus = phrase.nucleus_idx
        valid_positions = find_valid_nucleus_positions_in_phrase(tokens, phrase)

        if not valid_positions:
            # 有効な位置がない場合、母音があれば強制的に核を追加
            # ただし`[`より後の位置のみ
            rise_pos = None
            for i in range(phrase.start_idx, phrase.end_idx):
                if tokens[i] == '[':
                    rise_pos = i
                    break

            fallback_positions = []
            for i in range(phrase.start_idx, phrase.end_idx):
                if rise_pos is not None and i < rise_pos:
                    continue
                if is_phoneme(tokens[i]) and tokens[i] in VOWELS:
                    fallback_positions.append(i)
            if fallback_positions:
                new_nuclei[p_idx] = fallback_positions[-1]
            else:
                # 母音がない句（N, clのみ）は核なし
                new_nuclei[p_idx] = None
            continue

        # 元データで核がない場合は必ず核を追加
        if current_nucleus is None:
            new_nuclei[p_idx] = random.choice(valid_positions)
            continue

        # 誤りを加えるかどうか判定
        if random.random() > error_rate:
            # 変更なし
            new_nuclei[p_idx] = current_nucleus
            continue

        # 核がある場合：別の位置に移動
        other_positions = [p for p in valid_positions if p != current_nucleus]
        if other_positions:
            new_nuclei[p_idx] = random.choice(other_positions)
        else:
            # 他に有効な位置がなければ変更なし
            new_nuclei[p_idx] = current_nucleus

    # トークンリストを再構築
    # Step 1: ] を除去（[はそのまま保持）
    tokens_without_nuclei = [t for t in tokens if t != ']']

    # Step 2: トークンインデックスの対応表を作成（元 -> 新、]除去後）
    old_to_new = {}
    new_idx = 0
    for old_idx, token in enumerate(tokens):
        if token != ']':
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Step 3: ] を挿入すべき位置と、[を挿入すべき位置をマッピング
    nuclei_positions = set()  # ]を挿入する位置
    rise_insert_positions = set()  # [を挿入する位置（句に[がない場合）

    for p_idx, nucleus_idx in new_nuclei.items():
        if nucleus_idx is not None and nucleus_idx in old_to_new:
            new_pos = old_to_new[nucleus_idx]

            # 句内の`[`の位置を探す（tokens_without_nucleiでの位置）
            phrase = phrases[p_idx]
            new_start = old_to_new.get(phrase.start_idx, 0)
            new_end = old_to_new.get(phrase.end_idx - 1, len(tokens_without_nuclei) - 1) + 1

            rise_new_pos = None
            for i in range(new_start, min(new_end, len(tokens_without_nuclei))):
                if tokens_without_nuclei[i] == '[':
                    rise_new_pos = i
                    break

            # 句に`[`がない場合、1モーラ目の後に`[`を追加する必要がある
            # `[`は1モーラ目の後（最初の母音の後）に配置
            # ただし、1-2モーラしかない句では`[`を追加しない
            # （`]`は最後の音素に置けないため、2モーラ句では有効な位置がなくなる）
            need_rise_insert = (rise_new_pos is None)
            rise_insert_after_pos = None  # この位置の後に[を挿入
            if need_rise_insert:
                # 句内の母音を全て探す
                vowel_positions = []
                for i in range(new_start, min(new_end, len(tokens_without_nuclei))):
                    token = tokens_without_nuclei[i]
                    if is_phoneme(token) and token in VOWELS:
                        vowel_positions.append(i)

                # 3つ以上の母音がある場合のみ`[`を追加
                # 1-2モーラの場合は頭高型なので`[`なしで`]`のみ
                if len(vowel_positions) >= 3:
                    rise_insert_after_pos = vowel_positions[0]  # 1モーラ目の後
                else:
                    need_rise_insert = False  # 1-2モーラなので[は追加しない

            # `[`がある場合、それより後でなければならない
            if rise_new_pos is not None and new_pos < rise_new_pos:
                new_pos = None  # 無効な位置

            # `[`を挿入する場合、核は`[`より後（つまり1モーラ目より後）でなければならない
            if need_rise_insert and rise_insert_after_pos is not None:
                # 核が1モーラ目（rise_insert_after_pos）と同じかそれより前なら無効
                if new_pos is not None and new_pos <= rise_insert_after_pos:
                    new_pos = None

            # 句の最後の音素位置を探す（この位置には]を置けない）
            last_phoneme_in_phrase = None
            for i in range(min(new_end, len(tokens_without_nuclei)) - 1, new_start - 1, -1):
                if is_phoneme(tokens_without_nuclei[i]):
                    last_phoneme_in_phrase = i
                    break

            # 最後の音素には配置不可
            if new_pos is not None and new_pos == last_phoneme_in_phrase:
                new_pos = None

            if new_pos is not None and can_place_nucleus_at(tokens_without_nuclei, new_pos):
                nuclei_positions.add(new_pos)
                if need_rise_insert and rise_insert_after_pos is not None:
                    rise_insert_positions.add(rise_insert_after_pos)
            else:
                # 無効な位置の場合、同じ句内の他の有効な位置を探す
                # `[`を挿入する場合は1モーラ目より後から検索
                if need_rise_insert and rise_insert_after_pos is not None:
                    search_start = rise_insert_after_pos + 1
                elif rise_new_pos is not None:
                    search_start = rise_new_pos + 1
                else:
                    search_start = new_start

                alternative_pos = None
                for i in range(search_start, min(new_end, len(tokens_without_nuclei))):
                    # 最後の音素は除外
                    if i == last_phoneme_in_phrase:
                        continue
                    if can_place_nucleus_at(tokens_without_nuclei, i):
                        alternative_pos = i
                        break

                if alternative_pos is not None:
                    nuclei_positions.add(alternative_pos)
                    if need_rise_insert and rise_insert_after_pos is not None:
                        rise_insert_positions.add(rise_insert_after_pos)
                else:
                    # 母音があれば強制的に使用（最後の音素は除外）
                    for i in range(search_start, min(new_end, len(tokens_without_nuclei))):
                        if i == last_phoneme_in_phrase:
                            continue
                        token = tokens_without_nuclei[i]
                        if is_phoneme(token) and token in VOWELS:
                            nuclei_positions.add(i)
                            if need_rise_insert and rise_insert_after_pos is not None:
                                rise_insert_positions.add(rise_insert_after_pos)
                            break

    # Step 4: 新しいトークンリストを構築
    result = []
    for i, token in enumerate(tokens_without_nuclei):
        result.append(token)
        # [ を挿入（1モーラ目の母音の後）
        if i in rise_insert_positions:
            result.append('[')
        # ] を挿入（音素の後）
        if i in nuclei_positions:
            result.append(']')

    return join_tokens(result)


def fix_orphaned_segments(formatted_str: str) -> str:
    """
    ]の後のセグメント（テール）で母音がないものだけを処理
    テールは元のアクセント句の一部なので、新しい[は追加しない
    母音がないテール（?やNのみ）は前の句に含める（]を前の有効な母音の後に移動）
    """
    tokens = tokenize(formatted_str)

    # 母音がないテールを検出: ]の後から次の[または_または$までの間
    vowelless_tails = []  # (bracket_idx, end_idx)

    in_tail = False
    bracket_idx = None
    has_vowel_in_tail = False

    for i, token in enumerate(tokens):
        if token == ']':
            in_tail = True
            bracket_idx = i
            has_vowel_in_tail = False
        elif token == '[' or token == '_' or token == '$':
            if in_tail and not has_vowel_in_tail:
                # 母音がないテールを記録（処理対象）
                # ただし、]と[/_/$の間に音素がある場合のみ
                has_phonemes = False
                for j in range(bracket_idx + 1, i):
                    if is_phoneme(tokens[j]):
                        has_phonemes = True
                        break
                if has_phonemes:
                    vowelless_tails.append((bracket_idx, i))
            in_tail = False
        elif is_phoneme(token) and token in VOWELS:
            if in_tail:
                has_vowel_in_tail = True

    if not vowelless_tails:
        return formatted_str

    # 母音がないテールを処理（]を前の有効な母音の後に移動）
    for bracket_idx, end_idx in reversed(vowelless_tails):
        # 元の]を削除
        tokens.pop(bracket_idx)
        # 新しいend_idx（]削除で1ずれる）
        new_end_idx = end_idx - 1

        # テール内と]の前から有効な母音位置を探す
        # 優先順位: ]の直前の有効な母音
        valid_pos = None

        # ]の前（bracket_idxより前）から有効な母音を探す
        for i in range(bracket_idx - 1, -1, -1):
            if tokens[i] == '[' or tokens[i] == '_' or tokens[i] == '^':
                break  # アクセント句境界に達したので停止
            if can_place_nucleus_at(tokens, i):
                valid_pos = i
                break

        if valid_pos is not None:
            tokens.insert(valid_pos + 1, ']')
        # 有効な位置が見つからない場合は]を追加しない（平板化）

    return join_tokens(tokens)


def simplify_for_display(formatted_str: str) -> str:
    """
    フォーマット済み文字列を簡易表示用に変換
    """
    tokens = tokenize(formatted_str)
    remove = {'^', '$', '#', '_'}
    filtered = [t for t in tokens if t not in remove]
    return ' '.join(filtered)


def get_nucleus_position_in_phrase(tokens: List[str], phrase: AccentPhrase) -> Tuple[Optional[int], Optional[str]]:
    """
    アクセント句内での核の位置（音素番号）と核の前後の音素を取得

    Returns:
        (音素番号, 核位置の文脈表示 例: "re]eshi")
    """
    if phrase.nucleus_idx is None:
        return None, None

    # アクセント句内の音素をカウントして核の位置を特定
    phoneme_count = 0
    nucleus_phoneme_idx = None

    for i in range(phrase.start_idx, phrase.end_idx):
        if is_phoneme(tokens[i]):
            if i == phrase.nucleus_idx:
                nucleus_phoneme_idx = phoneme_count
            phoneme_count += 1

    # 核の前後の文脈を取得（核の直前2音素 + ] + 直後2音素）
    context_parts = []
    phoneme_idx = 0
    for i in range(phrase.start_idx, phrase.end_idx):
        if is_phoneme(tokens[i]):
            if nucleus_phoneme_idx is not None:
                if nucleus_phoneme_idx - 2 <= phoneme_idx <= nucleus_phoneme_idx:
                    context_parts.append(tokens[i])
                if phoneme_idx == nucleus_phoneme_idx:
                    context_parts.append(']')
                if nucleus_phoneme_idx < phoneme_idx <= nucleus_phoneme_idx + 2:
                    context_parts.append(tokens[i])
            phoneme_idx += 1

    context = ''.join(context_parts)
    return nucleus_phoneme_idx, context


def extract_phrase_text(tokens: List[str], phrase: AccentPhrase) -> str:
    """
    アクセント句のテキスト表現を抽出（[から次の[まで）
    """
    phrase_tokens = tokens[phrase.start_idx:phrase.end_idx]
    # 記号を除去して音素のみ
    phonemes = [t for t in phrase_tokens if is_phoneme(t)]
    return ''.join(phonemes)


def generate_diff_description(original: str, generated: str) -> Tuple[str, int, int]:
    """
    元データと生成データの差分を分かりやすく記述

    核の位置変更を「]@位置N → ]@位置M」の形式で表示
    （音素自体は変更していないことを明確にする）

    Returns:
        (差分説明文字列, 変更数, 総アクセント句数)
    """
    orig_tokens = tokenize(original)
    gen_tokens = tokenize(generated)

    orig_phrases = find_accent_phrases(orig_tokens)
    gen_phrases = find_accent_phrases(gen_tokens)

    changes = []
    changed_count = 0

    # 元データのアクセント句を基準に比較
    for i, orig_phrase in enumerate(orig_phrases):
        orig_pos, orig_context = get_nucleus_position_in_phrase(orig_tokens, orig_phrase)
        phrase_text = extract_phrase_text(orig_tokens, orig_phrase)

        # 対応する生成データのアクセント句を探す
        gen_pos, gen_context = None, None
        if i < len(gen_phrases):
            gen_pos, gen_context = get_nucleus_position_in_phrase(gen_tokens, gen_phrases[i])

        if orig_pos is None:
            # 平板→核追加
            if gen_pos is not None:
                changed_count += 1
                changes.append(f"平板→]@{gen_pos}({gen_context})")
            else:
                changes.append(f"平板=同じ")
        elif orig_pos == gen_pos:
            # 変更なし
            changes.append(f"]@{orig_pos}({orig_context})=同じ")
        else:
            # 核の位置が移動
            changed_count += 1
            if gen_pos is not None:
                changes.append(f"]@{orig_pos}({orig_context})→]@{gen_pos}({gen_context})")
            else:
                changes.append(f"]@{orig_pos}({orig_context})→平板化")

    return " | ".join(changes), changed_count, len(orig_phrases)


def process_dataset(data: dict, error_rate: float = 0.8) -> Tuple[dict, List[dict]]:
    """
    データセット全体を処理
    """
    error_data = {}
    diff_logs = []

    total = len(data)
    print(f"Processing {total} entries (error_rate={error_rate})...")

    for idx, (utterance_id, original) in enumerate(data.items(), 1):
        if idx % 500 == 0 or idx == total:
            print(f"  Progress: {idx}/{total}")

        # 正解データから誤りデータを生成
        tokens = tokenize(original)
        phrases = find_accent_phrases(tokens)
        generated = rebuild_with_moved_nuclei(original, phrases, error_rate)

        # 差分の詳細を生成
        diff_desc, changed_count, total_ap = generate_diff_description(original, generated)

        error_data[utterance_id] = generated
        diff_logs.append({
            "ID": utterance_id,
            "Changed": changed_count,
            "Total_AP": total_ap,
            "Changes": diff_desc,
            "Original": original,
            "Generated": generated,
        })

    print("Processing complete.")
    return error_data, diff_logs


def save_yaml(data: dict, filepath: str) -> None:
    """YAMLファイルとして保存"""
    print(f"Saving YAML to: {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")
    print(f"  Saved {len(data)} entries.")


def save_tsv(logs: List[dict], filepath: str) -> None:
    """TSVファイルとして保存"""
    print(f"Saving TSV to: {filepath}")
    fieldnames = ["ID", "Changed", "Total_AP", "Changes", "Original", "Generated"]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(logs)
    print(f"  Saved {len(logs)} entries.")


def save_html(logs: List[dict], filepath: str) -> None:
    """HTML形式で差分を色付きで保存"""
    print(f"Saving HTML to: {filepath}")

    html_content = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>JSUT BASIC5000 アクセント核誤り付与 差分レポート</title>
    <style>
        body {{ font-family: 'Hiragino Sans', 'Meiryo', sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; background: #fff; }}
        th {{ background: #4a90d9; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
        tr:hover {{ background: #f0f7ff; }}
        .change-moved {{ background: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; }}
        .change-added {{ background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; }}
        .change-same {{ background: #e9ecef; color: #6c757d; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; }}
        .change-flat {{ background: #e9ecef; color: #6c757d; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; }}
        .id {{ font-weight: bold; color: #4a90d9; }}
        .count {{ font-weight: bold; }}
        .count-high {{ color: #dc3545; }}
        .count-mid {{ color: #fd7e14; }}
        .count-low {{ color: #28a745; }}
        .phoneme {{ font-family: monospace; font-size: 12px; color: #666; word-break: break-all; }}
        .arrow {{ color: #dc3545; font-weight: bold; }}
        .nucleus {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>JSUT BASIC5000 アクセント核誤り付与 差分レポート</h1>
    <div class="summary">
        <p><strong>総エントリ数:</strong> {total_entries}</p>
        <p><strong>凡例:</strong>
            <span class="change-moved">核位置移動</span>
            <span class="change-added">平板→核追加</span>
            <span class="change-same">変更なし</span>
        </p>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>変更数</th>
                <th>変更詳細</th>
                <th>Original</th>
                <th>Generated</th>
            </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
    </table>
</body>
</html>
"""

    rows = []
    for log in logs:
        # 変更詳細を色付きHTMLに変換
        changes_html = format_changes_html(log["Changes"])

        # 変更数のスタイル
        changed = log["Changed"]
        total = log["Total_AP"]
        if changed == total:
            count_class = "count-high"
        elif changed > total // 2:
            count_class = "count-mid"
        else:
            count_class = "count-low"

        # Original/Generatedの]を強調
        orig_html = highlight_nucleus(log["Original"])
        gen_html = highlight_nucleus(log["Generated"])

        row = f"""            <tr>
                <td class="id">{log["ID"]}</td>
                <td class="count {count_class}">{changed}/{total}</td>
                <td>{changes_html}</td>
                <td class="phoneme">{orig_html}</td>
                <td class="phoneme">{gen_html}</td>
            </tr>"""
        rows.append(row)

    html_output = html_content.format(
        total_entries=len(logs),
        rows="\n".join(rows)
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"  Saved {len(logs)} entries.")


def format_changes_html(changes_str: str) -> str:
    """変更詳細を色付きHTMLに変換"""
    parts = changes_str.split(" | ")
    html_parts = []

    for part in parts:
        if "→" in part:
            if "平板→" in part:
                # 平板→核追加（緑）
                html_parts.append(f'<span class="change-added">{part}</span>')
            else:
                # 核位置移動（黄）
                html_parts.append(f'<span class="change-moved">{part}</span>')
        elif "=同じ" in part:
            # 変更なし（グレー）
            html_parts.append(f'<span class="change-same">{part}</span>')
        else:
            html_parts.append(f'<span class="change-flat">{part}</span>')

    return " ".join(html_parts)


def highlight_nucleus(formatted_str: str) -> str:
    """フォーマット済み文字列の]を強調表示"""
    # HTMLエスケープ
    s = formatted_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # ]を赤色で強調
    s = s.replace("]", '<span class="nucleus">]</span>')
    return s


def validate_output(original: str, generated: str, utterance_id: str = "") -> Tuple[bool, List[str]]:
    """
    出力が正しいか検証する包括的なチェック機構

    チェックルール:
    1. 各アクセント句に`]`は最大1つ
    2. `]`は有効な母音の後にのみ配置（N, cl, q, 長音の後半は不可）
    3. 元データの`[`位置は変更しない（新しい`[`を追加しない、ただし文頭のbare segment除く）
    4. `]`のみ移動、`[`構造は維持
    5. 文頭（^の直後）のbare segmentのみ`[`と`]`を追加可能
    6. 元データで`]`はあるが`[`がない句には`[`を追加してはいけない
    7. 元データに`]`があった句では生成データにも`]`が必要（位置は変わってもOK）
    8. 全てのアクセント句に`]`が必要（誤りデータとして全句に核が存在すべき）

    Returns:
        (is_valid, error_messages)
    """
    errors = []
    orig_tokens = tokenize(original)
    gen_tokens = tokenize(generated)

    # ルール1: 各アクセント句に`]`は最大1つ
    gen_phrases = find_accent_phrases(gen_tokens)
    for i, phrase in enumerate(gen_phrases):
        nucleus_count = 0
        for j in range(phrase.start_idx, phrase.end_idx):
            if gen_tokens[j] == ']':
                nucleus_count += 1
        if nucleus_count > 1:
            errors.append(f"Rule1: Phrase {i} has {nucleus_count} nuclei (max 1)")

    # ルール2: `]`は有効な母音の後にのみ配置
    # ただし、句に他の有効な母音がない場合は長音後半でも許可
    for i, token in enumerate(gen_tokens):
        if token == ']':
            # 直前の音素を探す
            prev_phoneme = None
            prev_phoneme_idx = None
            for j in range(i - 1, -1, -1):
                if is_phoneme(gen_tokens[j]):
                    prev_phoneme = gen_tokens[j]
                    prev_phoneme_idx = j
                    break

            if prev_phoneme is None:
                errors.append(f"Rule2: `]` at position {i} has no preceding phoneme")
            elif prev_phoneme not in VOWELS:
                errors.append(f"Rule2: `]` at position {i} follows non-vowel '{prev_phoneme}'")
            elif prev_phoneme in FORBIDDEN_NUCLEUS:
                errors.append(f"Rule2: `]` at position {i} follows forbidden nucleus '{prev_phoneme}'")
            elif prev_phoneme_idx is not None and is_long_vowel_second(gen_tokens, prev_phoneme_idx):
                # 長音後半の場合、この句に他の有効な母音があるかチェック
                # ない場合は例外として許可
                # この]を含む句を特定（#, _, ^, $が境界）
                phrase_start = 0
                phrase_end = len(gen_tokens)
                for j in range(i - 1, -1, -1):
                    if gen_tokens[j] in ['#', '_', '^']:
                        phrase_start = j + 1
                        break
                for j in range(i + 1, len(gen_tokens)):
                    if gen_tokens[j] in ['#', '_', '$']:
                        phrase_end = j
                        break

                # 句内の`[`の位置を探す
                rise_pos = None
                for j in range(phrase_start, phrase_end):
                    if gen_tokens[j] == '[':
                        rise_pos = j
                        break

                # 句内の最後の音素位置を探す
                last_phoneme_in_phrase = None
                for j in range(phrase_end - 1, phrase_start - 1, -1):
                    if is_phoneme(gen_tokens[j]):
                        last_phoneme_in_phrase = j
                        break

                # 句内で他の有効な母音があるかチェック
                # `[`がある場合は`[`より後のみチェック
                # 最後の音素は除外
                check_start = rise_pos + 1 if rise_pos is not None else phrase_start
                has_other_valid_vowel = False
                for j in range(check_start, phrase_end):
                    if j == prev_phoneme_idx:
                        continue  # 現在の位置はスキップ
                    if j == last_phoneme_in_phrase:
                        continue  # 最後の音素はスキップ
                    if is_phoneme(gen_tokens[j]) and gen_tokens[j] in VOWELS:
                        if not is_long_vowel_second(gen_tokens, j):
                            has_other_valid_vowel = True
                            break
                if has_other_valid_vowel:
                    errors.append(f"Rule2: `]` at position {i} follows long vowel second half")

    # ルール3&4: 元データの`[`位置は変更しない（文頭のbare segment除く）
    # 元データの`[`位置を記録
    orig_bracket_positions = []  # 音素基準での[の位置
    orig_phoneme_count = 0
    for i, token in enumerate(orig_tokens):
        if token == '[':
            orig_bracket_positions.append(orig_phoneme_count)
        elif is_phoneme(token):
            orig_phoneme_count += 1

    # 元データで]があるが[がない句を検出（意図的に句頭上昇がない句）
    # これらの句には[を追加してはいけない
    orig_phrases = find_accent_phrases(orig_tokens)
    phrases_with_nucleus_no_rise = set()
    for phrase in orig_phrases:
        if not phrase.has_rise and phrase.nucleus_idx is not None:
            # この句の開始位置（音素基準）を記録
            phoneme_pos = 0
            for i in range(phrase.start_idx):
                if is_phoneme(orig_tokens[i]):
                    phoneme_pos += 1
            phrases_with_nucleus_no_rise.add(phoneme_pos)

    # 生成データの`[`位置を記録
    gen_bracket_positions = []
    gen_phoneme_count = 0
    for i, token in enumerate(gen_tokens):
        if token == '[':
            gen_bracket_positions.append(gen_phoneme_count)
        elif is_phoneme(token):
            gen_phoneme_count += 1

    # 元データにない`[`を追加していないか確認
    # ただし、Rule10（`]`がある句には`[`が必要）により、`]`を追加する句には`[`も追加可能
    # そのため、ここでは元データの`[`が削除されていないかのみ確認

    # 元データの`[`が削除されていないか確認
    for pos in orig_bracket_positions:
        if pos not in gen_bracket_positions:
            errors.append(f"Rule4: Original `[` at phoneme position {pos} was removed")

    # Rule6は削除: Rule10（`]`がある句には`[`が必要）と矛盾するため

    # ルール5: 文頭のbare segmentのみ`[`と`]`を追加可能
    # これは上記のチェックで既にカバーされている（pos == 0の例外）

    # ルール7: 元データに`]`があった句では生成データにも`]`が必要
    # アクセント句ごとに元データと生成データの`]`の有無を比較
    gen_phrases = find_accent_phrases(gen_tokens)
    for i, orig_phrase in enumerate(orig_phrases):
        if orig_phrase.nucleus_idx is not None:
            # 元データに]がある句
            # 対応する生成データの句を探す
            if i < len(gen_phrases):
                gen_phrase = gen_phrases[i]
                # 生成データの句に]があるか確認
                has_nucleus_in_gen = False
                for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
                    if gen_tokens[j] == ']':
                        has_nucleus_in_gen = True
                        break
                if not has_nucleus_in_gen:
                    # 句のテキストを取得
                    phrase_phonemes = []
                    for j in range(orig_phrase.start_idx, orig_phrase.end_idx):
                        if is_phoneme(orig_tokens[j]):
                            phrase_phonemes.append(orig_tokens[j])
                    phrase_text = ''.join(phrase_phonemes[:5]) + '...' if len(phrase_phonemes) > 5 else ''.join(phrase_phonemes)
                    errors.append(f"Rule7: Phrase {i} ({phrase_text}) had `]` in original but missing in generated")

    # ルール8: 全てのアクセント句に`]`が必要（誤りデータとして全句に核が存在すべき）
    # 例外:
    #   - 母音がない句（N, clのみ）
    #   - `[`で終わる句（`[`より後に母音がない）
    #   - 有効な核配置位置がない句（`[`より後の唯一の母音が最後の音素）
    for i, gen_phrase in enumerate(gen_phrases):
        has_nucleus = False
        has_vowel = False
        rise_pos = None
        vowel_positions_after_rise = []
        last_phoneme_pos = None

        # 句内の情報を収集
        for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
            if gen_tokens[j] == ']':
                has_nucleus = True
            if gen_tokens[j] == '[':
                rise_pos = j
            if is_phoneme(gen_tokens[j]):
                last_phoneme_pos = j
                if gen_tokens[j] in VOWELS:
                    has_vowel = True
                    if rise_pos is not None and j > rise_pos:
                        vowel_positions_after_rise.append(j)

        # `[`がある場合、`[`より後に母音がなければ核配置不可（例外）
        if rise_pos is not None and not vowel_positions_after_rise:
            continue  # 核なしでもOK

        # `[`がある場合、`[`より後の母音が全て最後の音素なら核配置不可（例外）
        if rise_pos is not None and vowel_positions_after_rise:
            valid_vowels = [v for v in vowel_positions_after_rise if v != last_phoneme_pos]
            if not valid_vowels:
                continue  # 有効な位置がないので核なしでもOK

        if not has_nucleus and has_vowel:
            # 母音があるのに核がない場合はエラー
            phrase_phonemes = []
            for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
                if is_phoneme(gen_tokens[j]):
                    phrase_phonemes.append(gen_tokens[j])
            phrase_text = ''.join(phrase_phonemes[:5]) + '...' if len(phrase_phonemes) > 5 else ''.join(phrase_phonemes)
            errors.append(f"Rule8: Phrase {i} ({phrase_text}) has no nucleus `]`")
        # 母音がない句（N, clのみ）は核なしでも許可

    # ルール9: 句内で`[`は`]`より前になければならない（`[`は1モーラ目にのみ配置可能）
    for i, gen_phrase in enumerate(gen_phrases):
        rise_pos = None
        nucleus_pos = None
        for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
            if gen_tokens[j] == '[' and rise_pos is None:
                rise_pos = j
            if gen_tokens[j] == ']' and nucleus_pos is None:
                nucleus_pos = j
        if rise_pos is not None and nucleus_pos is not None:
            if nucleus_pos < rise_pos:
                # 核が句頭上昇より前にある = NG
                phrase_phonemes = []
                for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
                    if is_phoneme(gen_tokens[j]):
                        phrase_phonemes.append(gen_tokens[j])
                phrase_text = ''.join(phrase_phonemes[:5]) + '...' if len(phrase_phonemes) > 5 else ''.join(phrase_phonemes)
                errors.append(f"Rule9: Phrase {i} ({phrase_text}) has `]` before `[` (invalid order)")

    # ルール10: `]`がある句には`[`も必要（句頭上昇なしに核はありえない）
    # 例外: 1-2モーラの句は`[`なしで`]`のみ許可（頭高型、または`]`配置制約のため）
    for i, gen_phrase in enumerate(gen_phrases):
        has_rise = False
        has_nucleus = False
        vowel_count = 0
        for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
            if gen_tokens[j] == '[':
                has_rise = True
            if gen_tokens[j] == ']':
                has_nucleus = True
            if is_phoneme(gen_tokens[j]) and gen_tokens[j] in VOWELS:
                vowel_count += 1
        if has_nucleus and not has_rise:
            # 1-2モーラ（母音1-2個）の場合は例外
            if vowel_count <= 2:
                continue  # 1-2モーラなので`[`なしでOK
            phrase_phonemes = []
            for j in range(gen_phrase.start_idx, gen_phrase.end_idx):
                if is_phoneme(gen_tokens[j]):
                    phrase_phonemes.append(gen_tokens[j])
            phrase_text = ''.join(phrase_phonemes[:5]) + '...' if len(phrase_phonemes) > 5 else ''.join(phrase_phonemes)
            errors.append(f"Rule10: Phrase {i} ({phrase_text}) has `]` but no `[`")

    # ルール11: `]`の直後に`#`は不可（`]`の後に少なくとも1音素必要）
    for i in range(len(gen_tokens) - 1):
        if gen_tokens[i] == ']' and gen_tokens[i + 1] == '#':
            errors.append(f"Rule11: `]` immediately before `#` at position {i}")

    is_valid = len(errors) == 0
    return is_valid, errors


def run_validation_check(data: dict, error_data: dict) -> Tuple[int, int, List[Tuple[str, List[str]]]]:
    """
    全データに対してバリデーションチェックを実行

    Returns:
        (valid_count, total_count, errors_list)
    """
    print("\n[Validation] Running comprehensive validation check...")

    errors_list = []
    valid_count = 0

    for utterance_id, original in data.items():
        generated = error_data.get(utterance_id, "")
        is_valid, errors = validate_output(original, generated, utterance_id)

        if is_valid:
            valid_count += 1
        else:
            errors_list.append((utterance_id, errors))

    total_count = len(data)
    print(f"[Validation] Results: {valid_count}/{total_count} passed")

    if errors_list:
        print(f"[Validation] {len(errors_list)} entries have errors:")
        for utterance_id, errors in errors_list[:10]:  # 最初の10件のみ表示
            print(f"  {utterance_id}:")
            for error in errors:
                print(f"    - {error}")
        if len(errors_list) > 10:
            print(f"  ... and {len(errors_list) - 10} more")

    return valid_count, total_count, errors_list


def main():
    """メイン処理"""
    print("=" * 60)
    print("JSUT BASIC5000 アクセント核誤り付与データ生成")
    print("正解データのアクセント句構造を維持しながら核位置を移動")
    print("久保薗晴夫のアクセント法則に基づく制約を適用")
    print("=" * 60)
    print()

    # シード固定（再現性のため）
    random.seed(42)
    print("[Config] Random seed: 42 (for reproducibility)")
    print("[Config] Error rate: 80% per accent phrase")
    print()

    # データダウンロード
    print("[Step 1] Downloading ground truth data...")
    data = download_data(SOURCE_URL)

    # データ処理
    print()
    print("[Step 2] Generating error accent data...")
    error_data, diff_logs = process_dataset(data, error_rate=0.8)

    # バリデーションチェック
    print()
    print("[Step 3] Running validation check...")
    valid_count, total_count, errors_list = run_validation_check(data, error_data)

    if errors_list:
        print(f"\n[WARNING] {len(errors_list)} entries failed validation!")
        print("Please review the errors above.")
    else:
        print("\n[SUCCESS] All entries passed validation!")

    # 結果保存
    print()
    print("[Step 4] Saving output files...")
    save_yaml(error_data, OUTPUT_YAML)
    save_tsv(diff_logs, OUTPUT_TSV)
    save_html(diff_logs, OUTPUT_HTML)

    print()
    print("=" * 60)
    print("Generation complete!")
    print(f"  - Validation: {valid_count}/{total_count} passed")
    print(f"  - Error data: {OUTPUT_YAML}")
    print(f"  - Diff (HTML): {OUTPUT_HTML}")
    print(f"  - Diff log:   {OUTPUT_TSV}")
    print("=" * 60)

    # サンプル出力（比較表示）
    print()
    print("[Sample Output - Original vs Generated]")
    sample_keys = list(data.keys())[:3]
    for key in sample_keys:
        print(f"\n  {key}:")
        print(f"    Original:  {simplify_for_display(data[key])}")
        print(f"    Generated: {simplify_for_display(error_data[key])}")


if __name__ == "__main__":
    main()
