import shap


class ShapUtil:
    @staticmethod
    def get_shap_explainer(model_name, estimator, X):
        if model_name in ["RFR", "DTR", "GBR", "XGBR"]:
            explainer = shap.TreeExplainer(estimator, X)
        # elif model_name in ["EN"]:
        #     explainer = shap.LinearExplainer(estimator, X)
        elif model_name in ["EN", "SVR", "KNR"]:
            explainer = shap.Explainer(estimator.predict, X)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        return explainer
