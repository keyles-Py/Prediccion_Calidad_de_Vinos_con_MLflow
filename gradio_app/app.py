from google import genai
import gradio as gr
import mlflow
import pandas as pd
import joblib

gemini_api_key = "AIzaSyDSJRhRW_0VlPpukX3Qlqbjz3YMC8gu0z8"

client = genai.Client(api_key=gemini_api_key)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("winequality-linear-regression")

def explicar_prediccion(
    csv_filepath: str, 
    model_filepath: str = "../notebooks/modelo_rf.pkl"
) -> str:
    """
    Recibe la ruta del archivo temporal (dada por gr.File), lee el CSV con Pandas, 
    hace una predicción y usa la IA para generar una explicación textual.

    Args:
        csv_filepath (str): La ruta temporal del archivo CSV que proporciona Gradio.
        model_filepath (str): La ruta del archivo del modelo de Regresión Logística (.pkl).

    Returns:
        str: La explicación generada por la IA sobre la predicción, o un mensaje de error.
    """
    try:
        modelo = joblib.load(model_filepath)
    except FileNotFoundError:
        return f"Error: No se encontró el archivo del modelo en la ruta: {model_filepath}"
    except Exception as e:
        return f"Error al cargar el modelo: {e}"
    try:
        if csv_filepath is None:
             return "Error: No se ha subido ningún archivo CSV."
        df = pd.read_csv(csv_filepath, sep=";")
        
        if df.empty:
            return "Error: El archivo CSV está vacío."
        prep_data = df.iloc[0].to_frame().T 

    except FileNotFoundError:
        return f"Error al acceder al archivo temporal: {csv_filepath}"
    except Exception as e:
        return f"Error al leer o preprocesar el CSV: {e}"
    
    try:
        valor_predicho = modelo.predict(prep_data)[0]
        datos_string = prep_data.iloc[0].to_dict()
        datos_str_limpio = ", ".join([f"{k}: {v:.2f}" for k, v in datos_string.items()])
        
    except Exception as e:
        return f"Error durante la predicción con el modelo: {e}"
    
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("prediccion_calidad", valor_predicho)
            mlflow.log_text(datos_str_limpio, "input_features.txt")
            mlflow.log_artifact(csv_filepath, "raw_data")

            prompt_explicacion = (
                f"""Eres un enólogo experto. Basado en las siguientes características de un vino blanco:
                {datos_str_limpio} se predijo una calidad de {valor_predicho:.2f} en una escala de 0 a 10. Proporciona una explicación clara, breve y técnica de por qué este vino tiene esa calidad.
                No inventes datos. Sé objetivo y basado en atributos comunes de calidad (equilibrio, acidez, alcohol, etc.).
                Retorna la explicación en español.."""
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_explicacion,
            )
            explicacion = response.text

            if explicacion:
                explicacion2 = explicacion.strip()
            else:
                explicacion2 = "Error: El modelo de IA no devolvió contenido."
            
            mlflow.set_tag("genai_model", "gemini-2.5-flash")
            mlflow.set_tag("model_filepath_used", model_filepath) 
            mlflow.set_tag("prediction_model_type", "RandomForestRegressor")
            mlflow.log_text(explicacion2, "genai_explanation.txt")

            return explicacion2

    except Exception as e:
        return f"Error al generar la explicación con la IA: {e}"
    
iface = gr.Interface(
    fn=explicar_prediccion,
    inputs=gr.File(label="Sube el archivo CSV", type="filepath", file_count='single'), 
    outputs=[gr.Textbox(label="Explicación", lines=10)],
    title="Predicción de Calidad de Vino con Explicación IA"
)

iface.launch()