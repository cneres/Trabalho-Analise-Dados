import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np

class VisualizationHandler:
    def __init__(self):

        sns.set_theme()  

    def generate_visualizations(self, data):
        visuals = {}
        try:

            numeric_cols = data.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_cols = data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if numeric_cols:
                visuals.update(self._generate_correlation_heatmap(data, numeric_cols))
                visuals.update(self._generate_distributions(data, numeric_cols))
            if categorical_cols:
                visuals.update(self._generate_categorical_plots(data, categorical_cols))

            if self._has_geographic_data(data):
                visuals.update(self._generate_geographic_visualizations(data))

            return visuals

        except Exception as e:
            return {"error": f"Visualization error: {str(e)}"}

    def _has_geographic_data(self, data):
        """Check if dataset contains geographic information"""
        required_cols = {"latitude", "longitude"}
        return required_cols.issubset(data.columns)

    def _generate_correlation_heatmap(self, data, numeric_cols):
        if len(numeric_cols) < 2:
            return {}

        corr_matrix = data[numeric_cols].corr()

        epsilon = 1e-10
        mask = np.abs(corr_matrix) > epsilon

        mask = np.triu(mask, k=1)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix * mask, annot=True, cmap="coolwarm", fmt=".2f")
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
        plt.close()
        buffer.seek(0)

        relevant_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                var1, var2 = numeric_cols[i], numeric_cols[j]
                correlation = corr_matrix.iloc[i, j]

                if abs(correlation) <= epsilon or self._are_related_variables(
                    var1, var2
                ):
                    continue

                relevant_correlations.append(
                    {"var1": var1, "var2": var2, "correlation": correlation}
                )

        relevant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "correlation_heatmap": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "correlation_table": relevant_correlations,
        }

    def _are_related_variables(self, var1, var2):
        """Check if two variables are related based on their names"""

        var1, var2 = var1.lower(), var2.lower()

        if var1 in var2 or var2 in var1:
            return True

        patterns = [

            lambda x: x.replace("_norm", ""),
            lambda x: x.replace("_std", ""),
            lambda x: x.replace("_scaled", ""),
            lambda x: x.replace("_pct", ""),
            lambda x: x.replace("_ratio", ""),
            lambda x: x.replace("_log", ""),
            lambda x: x.strip("_0123456789"),  
        ]

        for pattern in patterns:
            if pattern(var1) == pattern(var2):
                return True

        common_prefix_length = 4
        if len(var1) >= common_prefix_length and len(var2) >= common_prefix_length:
            if var1[:common_prefix_length] == var2[:common_prefix_length]:
                return True

        return False

    def _generate_distributions(self, data, numeric_cols):
        
        visuals = {}
        for col in numeric_cols[:3]:  
            plt.figure(figsize=(8, 5))
            sns.histplot(data[col], kde=True)
            plt.title(f"Distribution: {col}")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
            plt.close()
            buffer.seek(0)

            visuals[f"distribution_{col}"] = base64.b64encode(buffer.getvalue()).decode(
                "utf-8"
            )

        return visuals

    def _generate_categorical_plots(self, data, categorical_cols):
        
        visuals = {}
        for col in categorical_cols[:3]:  
            plt.figure(figsize=(8, 5))
            value_counts = data[col].value_counts()
            if len(value_counts) > 20:  
                value_counts = value_counts.head(20)
            sns.barplot(x=value_counts.values, y=value_counts.index)
            plt.title(f"Category Distribution: {col}")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
            plt.close()
            buffer.seek(0)

            visuals[f"categorical_{col}"] = base64.b64encode(buffer.getvalue()).decode(
                "utf-8"
            )

        return visuals

    def _generate_geographic_visualizations(self, data):
        
        try:

            center = [data["latitude"].mean(), data["longitude"].mean()]
            m = folium.Map(location=center, zoom_start=12)

            marker_cluster = MarkerCluster().add_to(m)

            for idx, row in data.iterrows():
                popup_text = "<br>".join(
                    f"{col}: {row[col]}"
                    for col in data.columns
                    if col not in ["latitude", "longitude"]
                )
                folium.Marker(
                    location=[row["latitude"], row["longitude"]], popup=popup_text
                ).add_to(marker_cluster)

            heat_data = data[["latitude", "longitude"]].values.tolist()
            HeatMap(heat_data).add_to(m)

            map_html = m._repr_html_()

            return {
                "geographic_map": map_html,
            }

        except Exception as e:
            return {"geographic_error": str(e)}