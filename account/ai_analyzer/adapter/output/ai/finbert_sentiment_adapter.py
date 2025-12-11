import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ai_analyzer.application.port.sentiment_analysis_port import SentimentAnalysisPort


class FinbertSentimentAdapter(SentimentAnalysisPort):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            print("Loading KR-FinBERT-SC model...")

            model_name = "snunlp/KR-FinBert-SC"
            # 1. 모델과 토크나이저 로드
            cls.__instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.__instance.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # 2. 파이프라인 생성 (GPU가 있으면 device=0, 없으면 -1)
            device = 0 if torch.cuda.is_available() else -1
            cls.__instance.classifier = pipeline(
                "sentiment-analysis",
                model=cls.__instance.model,
                tokenizer=cls.__instance.tokenizer,
                device=device  # GPU 가속 사용
            )
            print(f"KR-FinBERT-SC model loaded. (Device: {device})")
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    # 메서드 이름을 analyze로 통일 (UseCase에서 호출하기 편하게)
    def analyze(self, text: str) -> dict:
        # 3. 파이프라인에서 truncation 옵션 사용 (토큰 단위 512개 제한)
        # 텍스트 길이 제한 (토큰 단위 권장)
        results = self.classifier(text, truncation=True, max_length=512)
        top_result = results[0]

        # 4. 딕셔너리 반환
        return dict(
            label=top_result['label'],  # 'positive', 'negative', 'neutral' 중 하나
            score=top_result['score']
        )