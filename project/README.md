# Rainforest Connection Species Audio Detection
###### tags: `Tag(kaggle)`
# 目的
25種の動物の種の中から鳴き声で何がいるか推定する

# 不明点
- label-weighted label-ranking average precisionの求め方
    - 何を重視したメトリックか
        - groud truthラベルの数によって重み付けされます?
        - テスト項目ごとに複数の真のラベルが存在する可能性がある場合の平均相互ランク尺度の一般化です。
        - ラベル加重部分は、全体のスコアがテストセット内のすべてのラベルの平均であり、各ラベルが等しい加重を受けることを意味します
        - 言い換えると、各テスト観測値は、観測値で見つかったグラウンドトゥルースラベルの数によって重み付けされます。
        - 各サンプルに関連付けられたラベルにより良いランクを付けることができれば、このパフォーマンス測定値は高くなります。
        - サンプルごとに関連するラベルが1つだけある場合、ラベルランキングの平均精度は、平均相互ランクと同等です。
        - ランクの効き方は予測したラベルの順位と正解の順位の位置の差により項が変化する
            - 正解1. Bird 2. Flog 3. incect
            - 予測1.Flog 2. incect 3. Bird
            - この場合はBirdは正解で1、予測で3い位置付けられるためこれを(正解ランク / 予測ランク)で項を計算する
                - 結果は1/3=0.333…となる
        - [Explaining the competition metric](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/206250)
    - [ Label ranking average precision](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision)
    - [ More on the Evaluation Metrics](https://www.kaggle.com/mrutyunjaybiswal/rainforest-tfrecords-with-audio-eda-metrics)
- 半教師あり学習をどの様に活用するか
    - 「貧弱な」ラベリングを利用して、現実の世界でうまく機能するモデルを構築することだと思います
- 同じ種でも注釈付きの呼び出しはすべて同じ周波数間隔を持っていない可能性があります
    - 個体差?
    - ばらつきの対応が必要かもしれない
    - 周波数間隔にばらつきがある理由は、一部の種では、同じコールタイプを発信できますが、周波数がわずかに異なるためです。
    - 対照的に、プレーンLRAPは各テスト観測に等しい重みを与えるため、観測に複数のラベルがある場合、個々のラベルの寄与が割り引かれます。
- 他の標的種または非標的種も記録に存在する可能性があります
    - 他の種を間違って推定しない工夫が必要
- train_tp, train_fpの活用について
    - カラムは同じ
    - レコード数はtpよりfpのほうが6倍多い
    - トレーニングを支援するための偽陽性ラベルの出現も含まれていることに注意してください
        - これがfpに当たる
        - is_tpカラムにてデータが元が識別できる
    - 
- 推定モデルは音を入力とするか、周波数画像を入力とするかで方向性が変わりそう
- 鳥コンペのソリューションも参考にしてみる

# Description
朝の鳥のさえずりやカエルの夕べの鳴き声を楽しまない人はいませんか？動物は甘い歌や自然の雰囲気だけではありません。熱帯雨林の種の存在は、気候変動や生息地の減少の影響を示す良い指標となります。これらの種は目で見るよりも耳で聞く方が簡単なので、地球規模で機能する音響技術を使用することが重要です。機械学習技術によって提供されるようなリアルタイムの情報は、人間が環境に与える影響を早期に発見することを可能にします。その結果、より効果的な保全管理の意思決定が可能になる可能性があります。

種の多様性や豊富さを評価する従来の方法は、コストがかかり、空間的にも時間的にも制限がある。また、ディープラーニングによる自動音響識別は成功しているが、モデルは種ごとに多数の訓練サンプルを必要とする。このため、保全活動の中心となる希少種への適用には限界があります。そこで、限られたトレーニングデータでノイズの多い音風景の中で高精度な種の検出を自動化する方法が解決策となります。

レインフォレスト・コネクション（RFCx）は、遠隔地の生態系を保護し、研究するための世界初のスケーラブルなリアルタイムモニタリングシステムを開発しました。ドローンや人工衛星のような視覚的な追跡システムとは異なり、RFCxは音響センサーに依存しており、年間を通して選ばれた場所で生態系のサウンドスケープを監視しています。RFCxの技術は、地域のパートナーが適応的管理の原則に基づいて野生生物の回復と回復の進捗状況を測定することを可能にする包括的な生物多様性モニタリングプログラムをサポートするために進歩してきました。また、RFCxモニタリングプラットフォームは、解析のための畳み込みニューラルネットワーク（CNN）モデルを作成する機能も備えています。

このコンテストでは、熱帯のサウンドスケープ録音から鳥やカエルの種を自動検出します。限られた音響的に複雑なトレーニングデータでモデルを作成します。鳥やカエルの音だけではなく、虫の音が1～2匹聞こえてくることが予想されますが、これはモデルがフィルタリングして除去する必要があります。

成功すれば、急速に拡大している科学分野、つまり自動化された環境音響モニタリングシステムの開発に貢献することができます。その結果、リアルタイムの情報が得られれば、人間の環境への影響を早期に発見できるようになり、環境保全をより迅速かつ効果的に行うことができるようになります。

# Evaluation
## ファイル形式
音声ファイル形式
単一の音声と複数音声のファイルが存在する
予測を行う際はオーディオファイルレベルで実行され開始・終了のタイムスタンプは必要ない
## メトリック
使うメトリックはラベル加重ラベルランク付けの平均精度(label-weighted label-ranking average precision)
テスト項目ごとに複数の真のラベルが存在する可能性がある場合の平均相互ランク測定値の一般化です。
ラベル加重」の部分は、全体のスコアがテストセットのすべてのラベルの平均であることを意味し、各ラベルは等しい重みを受け取ります（対照的に、プレーンラップは各テストオブザベーションに等しい重みを与え、それによって、オブザベーションが複数のラベルを持っている場合、個々のラベルの寄与を割引きます）。言い換えれば、各テストオブザベーションは、オブザベーションで見つかったgroud truthラベルの数によって重み付けされます。
## Submission
テストセットのrecording_idごとに、オーディオサンプルで各種のラベルが見つかる確率を予測する必要があります。
ファイルにはヘッダー（接頭辞がsの付いた各種番号）が含まれ、次の形式である必要があります。
```
recording_id,s0,...,s23
000316da7,0.1,....,0.3
003bc2cb2,0.0,...,0.8
etc.
```

## About data
このコンテストでは、多数の種の音を含むオーディオファイルが与えられます。あなたの課題は、各テストオーディオファイルについて、与えられた種のそれぞれがオーディオクリップの中で聴こえる確率を予測することです。トレーニングファイルには、種の識別と種が聴こえた時間の両方が含まれていますが、時間の定位はテストの予測には含まれていません。

トレーニングデータには、トレーニングを支援するための偽陽性ラベルの発生(train_fp.csv)も含まれています。

このデータは、Recording_id、audio_wav (16 ビット PCM フォーマットでエンコードされている)、label_info (trainのみ)を含む TFRecord フォーマットの競技データで、以下の列(Recording_id を引いたもの)の,-delimited 文字列を提供します。

columnsのtはそれぞれ開始/終了時間、fは信号の頻度
is_tpは真陽性(1),偽陽性(0)

音声ファイルの形式はFlac形式でwave形式に可逆可能な可逆圧縮ファイル形式
[「FLACファイル」って何ですか？](http://flac.aki.gs/bony/?page_id=820)

tfrecordもデータに含まれる。
[TFRecords と tf.Example の使用法](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecord_files_using_tfdata)

# librosa
- load
    - wavデータとサンプリングデータが取得できる
- melspectrogram
    - melspectrogramに音声データを変換する
    - スペクトログラムはフーリエ変換を掛けてからy軸を対数スケールに変換してデシベルに単位を変えたもの
    - メルスケールは周波数スケールの非線形変換を行った結果のもの
        - メル尺度は、メル尺度上で互いに等距離にある音が、互いに距離が等しいために人間にも「聞こえる」ように構成されています。
        - メル尺度は、トーンの知覚周波数を実際の測定周波数に関連付ける尺度です。
        - 人間の耳が聞くことができるものにより厳密に一致するように周波数をスケーリングします（人間は、より低い周波数での音声の小さな変化を識別するのに優れています）。このスケールは、人間を対象とした一連の実験から導き出されました。
    - スペクトログラムのy軸をメルスケールにしたものがmelspectrogram
    - [Getting to Know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
- [spectral_centroid](https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html?highlight=spectral_centroid)
    - マグニチュードスペクトログラムの各フレームは正規化され、周波数ビン全体の分布として扱われ、フレームごとに平均（重心）が抽出されます
- [frames_to_time](https://librosa.org/doc/latest/generated/librosa.frames_to_time.html?highlight=frames_to_time#librosa.frames_to_time)
    - countをtimeに変換する
- [spectral_rolloff](https://librosa.org/doc/latest/generated/librosa.feature.spectral_rolloff.html?highlight=spectral_rolloff)
    - ロールオフ周波数は、各フレームについて、このフレームのスペクトルのエネルギーの少なくともroll_percent（デフォルトでは0.85）がこのビンとその下のビンに含まれるようなスペクトログラムビンの中心周波数として定義される。これは、例えば、roll_percentを1（または0）に近い値に設定することによって、最大（または最小）周波数を近似するために使用することができます。
    - 信号の形状の尺度です。これは、全スペクトルエネルギーの指定されたパーセンテージ（たとえば85％）がそれを下回る周波数を表します
- [spectral_bandwidth](https://librosa.org/doc/latest/generated/librosa.feature.spectral_bandwidth.html?highlight=bandwidth#librosa.feature.spectral_bandwidth)
    - bandwidthを計算する
- [zero_crossings](https://librosa.org/doc/latest/generated/librosa.zero_crossings.html?highlight=zero_crossings#librosa.zero_crossings)
    - 信号が正から負またはその逆に変化する速度
- [mfcc](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html?highlight=mfcc#librosa.feature.mfcc)
    - mfccsを計算する
    - Deep Learningにおいては必要な情報が失われるためMFCCは使わずに、最後の計算ステップである離散コサイン変換を省いたメルスペクトラム(log-mel spectrum)が使われるそうです。MFCCは従来手法である隠れマルコフモデル、混合ガウスモデル、サポートベクターマシンで使われることが多いです。
    - [MFCC（メル周波数ケプストラム係数）入門](https://qiita.com/tmtakashi_dist/items/eecb705ea48260db0b62)
- [chroma_stft](https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html?highlight=chroma_stft)
    - 波形またはパワースペクトログラムからクロマトグラムを計算します。
    - スペクトル全体が音楽オクターブの12の異なる半音（またはクロマ）を表す12のビンに投影される、音楽オーディオの面白くて強力な表現です
- [trim](https://librosa.org/doc/latest/generated/librosa.effects.trim.html?highlight=trim#librosa.effects.trim)
    - オーディオ信号から先頭と末尾の無音をトリミングします。
- [hpss](https://librosa.org/doc/latest/generated/librosa.effects.hpss.html?highlight=hpss#librosa.effects.hpss)
    - オーディオの時系列をハーモニックコンポーネントとパーカッシブコンポーネントに分解します
    - 倍音は音の色を表す特徴です
    - 知覚衝撃波は音のリズムと感情を表します
- [beat_track](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html?highlight=beat_track#librosa.beat.beat_track)
    - 動的計画法のビートトラッカー。 
    - ビートは、1の方法に従って3段階で検出されます。
        - 発症強度を測定する
        - 発症相関からテンポを推定する
        - 推定テンポとほぼ一致する開始強度のピークを選択します
- 
サンプリングデータ
アナログ信号からデジタル信号へ変換するADコンバーターにおいて、1 秒間に実行する標本化（サンプリング）処理の回数のこと。
[サンプリングレート・サンプリング周波数 とは](https://www.liveon.ne.jp/glossary/wk/sampling_rate.html#:~:text=%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AA%E3%83%B3%E3%82%B0%E3%83%AC%E3%83%BC%E3%83%88%E3%81%AF%E3%80%81%E3%80%8C1%E7%A7%92%E9%96%93,%E6%84%9F%E3%81%98%E3%82%8B%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82)


# Discussion
- トレーニングデータに存在するすべてのターゲットサウンドがtrain_tp.csvで注釈されているわけではありません。注釈が付けられた部分は、それらがどれほどまれに呼び出すかに基づいて、種によって異なります。
- 一つのテストセットで最大24種の音が存在する
- 半教師あり学習
- 周波数境界f_minおよびf_maxも考慮すると、ラベル付けされた信号が別の種の呼び出しと重複する可能性は低くなります。
- 競技会のトレーニングデータは、基本的な検出アルゴリズムを使用して収集されました。ここでは、各種/曲の種類の例が、大量のサウンドスケープオーディオと相互相関しています。
- 一致する可能性のあるものにはフラグが付けられ、専門家によってレビューされました。
- train_tp.csvの各行は、トレーニングオーディオファイルの1つのフラグ付きセグメントに対応し、専門家によって、species_idとsongtype_idが含まれていることが確認されました。
- train_fp.csvの各行は、種とソングタイプが含まれていないことが確認されたフラグ付きセグメントに対応しています。
- このコンペティションのアイデアは、「貧弱な」ラベリングを利用して、現実の世界でうまく機能するモデルを構築することだと思います
- またはいくつかの種/歌の種類、いくつかの周波数変動があり、注釈付きの呼び出しはすべて同じ周波数間隔を持っていない可能性があります
- 一部の種では、train_tp.csvに2つのソングタイプのデータがありますが、train_fp.csvには2つのうちの1つのデータしかありません。
- train_fp.csvに固有の周波数間隔はありません（train_tp.csvにはありません）。
- 最小および最大の周波数は音声分析に基づくのではなく、専門の生物学者の注釈に基づいています。
- 周波数間隔にばらつきがある理由は、一部の種では、同じコールタイプを発信できますが、周波数がわずかに異なるためです。
- 他の標的種または非標的種も記録に存在する可能性があります。 train_tp.csvですべてのターゲット呼び出しに注釈を付ける必要があるわけではありません
- すべてのテストファイルで対象種の有無がラベル付けされている
- ラベルのないオーディオにはターゲット種は存在しません
- 誤検出セグメントは、前述の相互相関アルゴリズムによって検出されたため、はい、ターゲットの種/曲タイプに対して少なくとも弱い正の相関があります
- FFTのみだと情報が落ちる主に時系列的な箇所そこでスペクトログラムを使うことでSRが変わることはあるが時系列情報を獲得することができると考えられる
- 以下アプローチで0.9以上を達成することができた
    - PANNs SED architecture
    - BCE Loss
    - Random 10sec clip
    - Using only tp file
    - MixUp
    - The base model is EfficientNet
- 以下の方法はうまく適用できなかった
    - Dice Loss
    - SpecAugment
    - Using fp file as background noise sound(For example, mix 0.9tp + 0.1fp)
    - Using fp file as no label sound (all zero label)
    - Self Supervised Learning (COLA)→論文上ではバッチは1024で環境制限で64にするとうまく動かない
    - Discover missing labels and retrain→tpファイルに不足ラベルが散見しているがそれを使って再トレーニングしても改善されなかった
- CVの組み方次第でSEDモデルでも不安定な結果を返してしまう
- 大きいベースネットほどパフォーマンスの低下が見られる
    - EfficientNetB0とMobileNetV2は良好なパフォーマンスだった
- 5kfoldを使った
- このコンテストのサンプル数は比較的少ないようです。トレーニング前のタスクで適切な埋め込みを作成すると役立つ場合があります。
- COLAが役立つかもしれない
    - 汎用音声表現学習のための新しい自己教師あり学習方法です。
    - https://arxiv.org/abs/2010.10915
- train_tp内のファイルで一つしかラベリングされていないファイルでも疑似ラベル(COLA)を使うと多くのラベルが発見される
- sklearnの提供するLRAPの計算は今回のコンペのものとは異なる 実装して修正した(pytorch)→[Correct Metric in Pytorch: LWLRAP](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418)
- mel-spectrogramは以下3つの要素で構成される
    - FFT (高速フーリエ変換、別名離散フーリエ変換) これは、信号の各周波数の強さを教えてくれますが、周波数が時間の経過とともにどのように変化するかは教えてくれません。
    - STFT (The Short Time Fourier Transform) FFTは、スライディングウィンドウの方法で時間軸に沿って信号の小さな塊に複数回適用され、周波数の強さが時間の経過とともにどのように変化するかを教えてくれます。
    - Mel Scale Transform (メル・スケール変換)。周波数をヘルツからメルに変換します。メル・スケールは、人間が音程の違いをどのように知覚するかを近似することを目的としています。
    - Mel Spectrogram = STFT + Mel Scale Transform.
    - [Beginner Help](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/199048)
- Birdcall identificationのソリューション→[Cornell Birdcall Identification - top rank notebooks and discussions](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197873)
- 外部データ対策として鳴き声による種別な隠蔽している→外部データを使うこと自体問題は無い
- 鳴き声のタイプが異なるデータを2種以上持つ種もいるためそれらを含めて推定する必要あり
- 音声検出のテクニック
    - RNNにより時系列的解く方法あるがあまり活用されているシーンは無い
    - MFCCsに変換後画像としてNNに入力してCVで問題を解く方法が多く使われている
- [Papers on Soundscapes and Animal Sound detection](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197874)
- Tensorflowを使ったData Augmentation→[🐧 🐦🦉 🦇 Tensorflow-io -Audio Augmentation and data preparation](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/201809)
- データ変換方法
    - Frequency masking
    - Time masking
    - White, pink noise injection
    - Pitch shifting
        - Only between Tmin and Tmax
        - Only shifting frequencies between Fmin and Fmax, and shifting them within a 95% confidence interval of all the Fmins and Fmaxes for that specific species_id
    - Selecting random N second crops of the audio, where N is determined species by species
        - Using a 10 second crop may be too large for songs that only last 1 second on average, for example.
    - reverberation

## Ref
- [train_tp vs. train_fp](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197866)
- [Methods that worked / did not work.](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/208830)
- [COLA: Contrastive learning of general purpose audio representations](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197805)
- [This is a missing label.](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209040)
- [🐤🦆 🦅🐸 Extensive Resource Compilation for Audio Data](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/199619)

# Notebooks memos
- librosaという音声系操作ライブラリがある
- IPythonにAudioというJupyter上でAudioを流せるクラスがある
- スペクトログラムの強度が強いところほどヒートマップ上で明るくなる
- species_idはtp,fp共に23, 17が高頻度で検出されておりその他は大体同等となっている
- songtype_idは1,4のみかつ1が85%以上
    - songtype_idの1の周波数は0-14000付近で4は1500-11000付近
- 鳴き声発生は種別の収録タイミングによらないが全体でピークが10秒もしくは50秒程度によく分布している
- 周波数と鳴き声の時間が特徴を持っている
    - [EDA and spectral signatures of species](https://www.kaggle.com/andregoios/eda-and-spectral-signatures-of-species)
- 自動音響識別システムはあるが種毎に多数のデータが必要
    - 今回のコンペでは希少種の識別が目的のためデータ数が少なくなる
    - 少ないデータパフォーマンスの良いモデルが求められる
- 種は鳥、カエル限定
- 目的の種別以外の音が多く入り込んでいる
- オーディオデータによって単一種のもの複数種のものが存在する
- 24種、9000ファイル
- train_tp, train_fpのcsvファイルの構造は共にレコーディングIDが同一でspecies_idが複数存在する状態となっている
- 出力する形式は多クラス予測
- songtype_idの4は16(1なし),17,23となっているそれ以外は1のみ
- フーリエ変換
    - 音声ファイルから必要な周波数を切り取るのに有用
    - 時間表現は周波数を曖昧にし、周波数表現も周波数を曖昧にします。
    - Short-time Fourier transform(STFT)
        - ウィンドウ化された信号のフーリエ変換のシーケンスです。
        - 信号の周波数成分が時間とともに変化する状況の時間ローカライズされた周波数情報を提供しますが、標準のフーリエ変換は、信号の時間間隔全体にわたって平均化された周波数情報を提供します。
        - 元の信号からの信号の固定長ウィンドウ。各ウィンドウにフーリエ変換を適用してから、すべてのウィンドウの合計を取得します。
        - それでも、ウィンドウの長さ、ウィンドウの形状、ウィンドウフィルター、処理などの選択にはあいまいさが残っています
- Spectral Centroids
    - 音の重心を表し音の明るさの尺度を与えます。スペクトルフレームの個々のセントロイドは、振幅で加重された平均周波数を振幅の和で割ったものとして定義されます。
- Cepstrum
    - スペクトルバンドの変化率の情報。
    - 
## Data Augmentation
- AddGaussianNoise
    - Gaussian Noiseを加えることでノイズが比較的少ないデータで学習して、ノイズが多い環境の音を処理しなければいけない場合などに汎化性能をあげることができます。
- GaussianNoiseSNR
    - GaussianNoiseでは元の信号が微弱なときに雑音に覆い隠されてしまうことがありうる
    - これを防ぐために元の音の中の信号の振幅を元に適切な雑音レベルを適応的に設定できるようにしたほうが使いやすい
        - 信号の大きさと雑音の大きさの比を表したものをSignal-to-Noise Ratio(SNR)
        - 信号の大きさ、といった場合には振幅のことをさすことが多いのですが多くの場合SNRは実際の振幅の比に対数を取ったものとして表現され
        - $SNR=20log_{10}*(\frac{A_{signal}}{A_{noise}})$
        - この量は大きければ大きいほど信号が強い、すなわち音が聞こえやすいことを表す量で単位はdB(デシベル)で表現されます。0dBで信号の強さと雑音の強さが釣り合っている状態で、負の場合には雑音の方が強い状態、正の場合には信号の方が強い状態です。
        - また、信号音の強さの推定法はいくつかあるかと思いますが、今回はクリップ内の振幅の絶対値の最大値を信号の振幅として扱います。
- PinkNoiseSNR
    - Pink Noiseは低周波数帯から低周波数帯にかけて徐々にノイズの強さが減少するようなノイズのことをさします。自然界に存在するノイズはこのようなノイズであるとされます。
- PitchShift
    - 音のピッチ(高低)に関する調整を施すData Augmentationで、効果として聞こえる音が高く/低くなります。メルスペクトログラム上では、パターンのある周波数帯が上または下にズレます。
- TimeStretch
    - 元の音を時間的に引き延ばしたり圧縮したりします。結果として音のスピードが速くなったり遅くなったりします。
- TimeShift
    - 時間的に音イベントをずらすような操作です。ズラした結果、元の音クリップの長さからはみ出した部分の扱いに関しては、前(または後ろ)に持っていってくっつける、無視して捨ててしまう、などのやり方があります。
- VolumeControl
    - 音量を調節します。音の認識には音量そのものよりSNRが影響するという話を以前紹介したかと思いますが、音量を調節することでメルスペクトログラムにはごく僅かな変化が生じます。また、音量をサイン曲線、コサイン曲線などに合わせて調節する、などはメルスペクトログラムには大きな変化をもたらすため有用です。

## Ref
- Data Augmentation参考[RFCX: Audio Data Augmentation(Japanese+English)](https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english)
- [Rainforest | TFRecords with Audio | EDA | Metrics](https://www.kaggle.com/mrutyunjaybiswal/rainforest-tfrecords-with-audio-eda-metrics)


# Model
- スペクトログラム図の画像解析
    - ResNET
- SED(Sound Event Detection)
    - [Introduction to Sound Event Detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)
    - [Methods that worked / did not work](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/208830)
- RFCｘ
    - CNNを使った音響モニタリングモデル
- model例
```
model: resnet34
augmentation: No
duration: trainied with 60 sec clip
channel: 1
data: only TP
epoch: 10 (best: 6)
split: 80/20
fold: 1

CV - 0.576
LB - 0.546
```

# 類似コンペ
- [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition)
- [Freesound Audio Tagging 2019](https://www.kaggle.com/c/freesound-audio-tagging-2019)
- [DCASE challenges](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197737)

## Ref
- [Previous Audio Competitions](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/197737)

# Ref
- [Audio Signal Processing for Machine Learning](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
- [introduction to audio content analysis](https://www.audiocontentanalysis.org/teaching/)
- [The dummy’s guide to MFCC](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)
- [Resources for Audio Event Detection & Scene Analysis w/ ML](https://www.ak.tu-berlin.de/fileadmin/a0135/downloads/resources_aed4dl.pdf)