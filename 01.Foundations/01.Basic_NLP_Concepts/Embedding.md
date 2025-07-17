# FastText

### Part1. 모든 것의 시작, “Word2Vec”

FastText를 이해하려면, 먼저 그 뿌리가 되는 **Word2Vec**을 알아야 합니다. 2013년 구글에서 개발한 Word2Vec은 "단어의 의미는 그 주변 단어에 의해 결정된다"는 아주 단순하면서도 강력한 아이디어에 기반합니다.

- **핵심 아이디어**: "강아지가 **멍멍** 짖는다"와 "고양이가 **야옹** 운다"라는 문장에서 '멍멍'과 '야옹'은 비슷한 위치에 등장하므로, 의미적으로도 유사한 벡터를 갖게 될 것이라고 추론하는 방식입니다.

Word2Vec은 이 아이디어를 구현하기 위해 두 가지 대표적인 학습 아키텍처를 사용합니다.

**1. CBOW (Continuous Bag-of-Words)**

**"주변 단어로 중심 단어 맞추기"**`[오늘, ___, 정말, 좋다]`라는 문맥이 주어졌을 때, 가운데 빈칸에 들어갈 단어가 '날씨'일 것이라고 예측하며 학습합니다. 여러 주변 단어 정보를 압축하여 사용하므로 학습 속도가 빠릅니다.

**2. Skip-gram**

**"중심 단어로 주변 단어 맞추기"**
CBOW와는 반대로, '날씨'라는 중심 단어가 주어졌을 때, 그 주변에 `[오늘, ___, 정말, 좋다]`가 올 것이라고 예측하며 학습합니다. 하나의 단어로 여러 단어를 예측해야 하므로, 단어의 의미를 더 풍부하게 학습할 수 있어 일반적으로 성능이 더 좋다고 알려져 있습니다.

→ 결론적으로는 “Skip-gram” 방식이 성능이 더 좋다고 알려져 있습니다. 중심 단어 하나로 여러 주변 단어를 예측해야 하므로, **특히 대규모 데이터셋에서 단어의 풍부한 의미적, 문맥적 관계를 학습하는 데 더 효과적이기 때문**입니다.

**Word2Vec의 명확한 한계**

Word2Vec은 획기적이었지만, 치명적인 단점이 있었습니다. 바로 **OOV(Out-of-Vocabulary)** 문제입니다. 훈련 데이터에 없었던 새로운 단어, 예를 들어 신조어 '최애'나 오타 '솨과'가 등장하면 Word2Vec은 이 단어의 벡터를 만들어내지 못하고 완전히 무시해버립니다.

### Part2. OOV를 해결하는 “FastText”

2016년 페이스북 AI 연구소에서 개발한 **FastText**는 바로 이 OOV 문제를 해결하기 위해 등장했습니다.

- **핵심 아이디어**: "단어를 통째로 보지 말고, 내부의 문자 조각들(characters n-gram)로 나눠보자!"

Word2Vec이 'apple'을 하나의 통단어로 보는 반면, FastText는 'apple'을 더 작은 단위로 분해합니다. 예를 들어 n을 3~6 으로 설정하면, `<ap`, `app`, `ppl`, `ple`, `le>` 와 같은 문자 n-gram(subword)들의 집합으로 표현됩니다. 그리고 중요한 것은 FastText는 이 subword 집합에 **원래 단어 'apple' 자체도 함께 포함하여 학습**한다는 점입니다.

**n-gram 벡터의 합, 어떻게 계산될까?**

'먹었다'라는 단어를 예로 들어보겠습니다. (n-gram 범위를 3~6으로 가정)

1. **분해**: 먼저, 단어의 시작과 끝을 표시하기 위해 특별 문자(e.g., `<` , `>`)를 붙여 `<먹었다>`로 만듭니다. 그 후, 정해진 길이(3~6)의 n-gram 조각으로 모두 분해합니다.
    - (n=3) `<먹었`, `먹었`, `었다`, `다>`
    - (n=4) `<먹었다`, `먹었다>`
    - (n=5) `<먹었다>`
    - (n=6) `<먹었다>`
    - 그리고 단어 전체인 `<먹었다>` 자체도 이 집합에 포함됩니다.
2. **조회**: 모델은 이미 학습 과정에서 이 모든 문자 n-gram 조각들의 벡터를 가지고 있습니다. 분해된 조각들(`<먹었`, `었다`, `<먹었다>` 등)의 벡터를 각각 조회합니다.
3. **합산 (또는 평균)**: 조회된 모든 벡터들을 간단히 **더하거나(sum) 평균을 내어(average)** '먹었다'라는 단어의 최종 벡터를 생성합니다.

*<참고: 실제 구현에서는 중복을 제거한 고유한 n-gram 집합을 사용합니다.>*

이 간단하지만 강력한 아이디어 덕분에 FastText는 놀라운 장점들을 갖게 됩니다.

- **OOV 문제 해결**: 훈련 데이터에 '최애'라는 단어가 없었어도, '최'와 '애'라는 글자 n-gram은 다른 단어들을 통해 이미 학습되었을 가능성이 높습니다. FastText는 이 조각들의 벡터를 조합하여 '최애'의 의미를 유추해냅니다.
- **오타에 강함**: '솨과'는 처음 보는 단어지만, '과'라는 n-gram이 '사과'와 겹치기 때문에 의미적으로 유사한 벡터를 생성할 수 있습니다.
- **형태학적으로 풍부한 언어(한국어)에 최적화**: '먹다', '먹고', '먹으니', '먹었다'는 모두 '먹'이라는 핵심 n-gram을 공유합니다. 따라서 FastText는 이 단어들이 문법적, 의미적으로 매우 가깝다는 것을 쉽게 학습합니다.

### Part 3: “FastText”의 구조: Shallow Neural Network

단 하나의 Hidden Layer 만 갖습니다.

FastText의 학습 아키텍처(Skip-gram 기준)는 다음과 같이 동작합니다.

1. **입력층 (Input Layer)**: 중심 단어(예: '날씨')를 구성하는 모든 subword 벡터들의 **평균 벡터**가 입력으로 들어갑니다.
2. **히든층 (Hidden Layer)**: 이 입력 벡터는 **가중치 행렬 W** 와 곱해집니다. 이 히든층이 바로 우리가 얻고 싶어하는 **'단어 벡터(임베딩)'를 학습하는 공간**입니다. 즉, 이 가중치 행렬 W가 바로 subword 벡터들을 저장하는 임베딩 행렬이 됩니다.
3. **출력층 (Output Layer)**: 히든층을 거친 결과값은 다시 출력 가중치 행렬 $W'$와 곱해진 후, 소프트맥스(Softmax) 함수를 통과합니다. 이를 통해 주변 단어(예: '오늘', '정말', '좋다')가 나타날 확률을 계산합니다.

**핵심은 "주변 단어를 잘 예측하는 방향으로 subword 벡터들을 업데이트하는 것"** 입니다. 이 예측 과정에서 오차가 발생하면, 역전파(Backpropagation)를 통해 히든층의 subword 벡터 값들을 계속해서 미세 조정합니다. 이 과정을 수많은 단어에 대해 반복하면, 문맥적 의미를 잘 담아내는 subword 벡터들이 만들어지는 것입니다.

### Part4. 실습

1. **Word2Vec**
    
    ```python
    from gensim.models import Word2Vec
    
    # 간단한 문장 데이터
    sentences = [
        ['나는', '어제', '밥을', '먹었다'],
        ['친구는', '오늘', '밥을', '먹는다']
    ]
    
    # Word2Vec 모델 학습
    w2v_model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=1)
    
    # '먹었다'와 유사한 단어 찾기
    print("Word2Vec '먹었다' 유사도:", w2v_model.wv.most_similar('먹었다'))
    
    # OOV 단어 테스트
    try:
        w2v_model.wv.most_similar('먹었니')
    except KeyError as e:
        print(f"Word2Vec OOV 테스트: {e}")
    ```
    
2. **FastText**
    
    ```python
    from gensim.models import FastText
    
    # 동일한 문장 데이터
    sentences = [
        ['나는', '어제', '밥을', '먹었다'],
        ['친구는', '오늘', '밥을', '먹는다']
    ]
    
    # FastText 모델 학습 (n-gram 범위 설정 가능)
    ft_model = FastText(sentences, vector_size=100, window=3, min_count=1, sg=1,
                        min_n=3, max_n=6) # 3~6글자의 n-gram 사용
    
    # '먹었다'와 유사한 단어 찾기
    print("\nFastText '먹었다' 유사도:", ft_model.wv.most_similar('먹었다'))
    
    # OOV 단어 테스트 (오류 없이 실행됨)
    print("FastText OOV 테스트 '먹었니' 유사도:", ft_model.wv.most_similar('먹었니'))
    ```
    
3. **모델 세부 내용 확인**
    
    ```python
    # 3-1. 어휘 사전(Vocabulary) 확인
    # model.wv는 학습된 단어 벡터와 사전을 담고 있는 객체입니다.
    vocab = model.wv.key_to_index
    print(f"1. 학습된 총 단어 수: {len(vocab)}개")
    print(f"   샘플 단어: {list(vocab.keys())[:10]}\n")
    
    # 3-2. 임베딩 행렬(Embedding Matrix) 확인
    # 이것이 바로 신경망의 '가중치(weights)'에 해당하는 부분입니다.
    embedding_matrix = model.wv.vectors
    print(f"2. 임베딩 행렬의 형태: {embedding_matrix.shape}")
    print("   => (총 단어 수, 벡터 차원)\n")
    
    # 3-3. 특정 단어의 벡터 확인
    sample_word = '밥을'
    word_vector = model.wv[sample_word]
    print(f"3. 단어 '{sample_word}'의 벡터 형태: {word_vector.shape}")
    print(f"   '{sample_word}' 벡터 값 (앞 5개만): {word_vector[:5]}\n")
    
    # 3-4. 학습 시 사용된 하이퍼파라미터 확인
    print("4. 모델 하이퍼파라미터 정보")
    print(f"   - 벡터 차원 (vector_size): {model.vector_size}")
    print(f"   - 윈도우 크기 (window): {model.window}")
    print(f"   - 학습 방식 (sg): {'Skip-gram' if model.sg else 'CBOW'}")
    print(f"   - 최소 단어 빈도 (min_count): {model.min_count}")
    ```
    

<aside>
⭐

위 코드를 실행하면, Word2Vec은 '먹었니'라는 OOV 단어에 대해 `KeyError`를 발생시키는 반면, FastText는 '먹었다', '먹는다'와 유사하다고 성공적으로 추론해내는 것을 명확히 확인할 수 있습니다.

</aside>