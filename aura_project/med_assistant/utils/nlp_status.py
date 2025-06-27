class NLPStatus:
    def __init__(self, classification, drbert, t5, whisper):
        self.classification_available = classification
        self.drbert_available = drbert
        self.t5_available = t5
        self.whisper_available = whisper

    @property
    def pipeline_label(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "Pipeline complète active"
        elif self.classification_available and self.drbert_available:
            return "Classification + DrBERT connectés, T5 en simulation"
        elif self.classification_available and self.t5_available:
            return "Classification + T5 connectés, DrBERT en simulation"
        elif self.drbert_available and self.t5_available:
            return "DrBERT + T5 connectés, classification en simulation"
        elif self.classification_available:
            return "Classification connectée, DrBERT et T5 en simulation"
        elif self.drbert_available:
            return "DrBERT connecté, classification et T5 en simulation"
        elif self.t5_available:
            return "T5 connecté, classification et DrBERT en simulation"
        return "Mode simulation complet"

    @property
    def description_label(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "Classification + entités + résumés réels"
        elif self.classification_available and self.drbert_available:
            return "Classification + entités réelles, résumés simulés"
        elif self.classification_available and self.t5_available:
            return "Classification + résumés réels, entités simulées"
        elif self.drbert_available and self.t5_available:
            return "Entités + résumés réels, classification simulée"
        elif self.classification_available:
            return "Classification réelle, entités et résumés simulés"
        elif self.drbert_available:
            return "Entités réelles, classification et résumés simulés"
        elif self.t5_available:
            return "Résumés réels, classification et entités simulées"
        return "Traitement local avec simulation complète"

    @property
    def border_class(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "border-green-500"
        elif self.classification_available or self.drbert_available or self.t5_available:
            return "border-yellow-500"
        return "border-red-500"

    @property
    def bg_class(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "bg-green-100"
        elif self.classification_available or self.drbert_available or self.t5_available:
            return "bg-yellow-100"
        return "bg-red-100"

    @property
    def text_class(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "text-green-600"
        elif self.classification_available or self.drbert_available or self.t5_available:
            return "text-yellow-600"
        return "text-red-600"

    @property
    def text_class_title(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "text-green-800"
        elif self.classification_available or self.drbert_available or self.t5_available:
            return "text-yellow-800"
        return "text-red-800"

    @property
    def pipeline_mode(self):
        if self.classification_available and self.drbert_available and self.t5_available:
            return "Pipeline complète"
        elif self.classification_available or self.drbert_available or self.t5_available:
            return "Hybride"
        return "Local"
