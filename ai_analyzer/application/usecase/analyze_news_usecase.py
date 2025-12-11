from ai_analyzer.application.port.sentiment_analysis_port import SentimentAnalysisPort
from ai_analyzer.domain.value_object.analysis_result_vo import AnalysisResultVO


class AnalyzeNewsUseCase:
    def __init__(self, sentiment_port: SentimentAnalysisPort):
        self.sentiment_port = sentiment_port

    def analyze(self, content: str) -> AnalysisResultVO:
        # 3. 감성 분석 수행
        sentiment_result = self.sentiment_port.analyze(content)

        return AnalysisResultVO(
            sentiment_label=sentiment_result['label'],
            sentiment_score=sentiment_result['score']
        )