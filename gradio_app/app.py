from openai import OpenAI
import gradio as gr
import mlflow

gemini_api_key = "AIzaSyDSJRhRW_0VlPpukX3Qlqbjz3YMC8gu0z8"

client = OpenAI(
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
  api_key=gemini_api_key,
)

def explicar_prediccion_gradio(
    df: pd.DataFrame, 
    model_filepath: str = "modelo_logreg.pkl"
) -> str:
    """
    Recibe un DataFrame de Pandas (típicamente desde Gradio), hace una predicción 
    de calidad con un modelo de Regresión Logística preentrenado y utiliza un modelo 
    de IA para generar una explicación textual de la predicción.

    Args:
        df (pd.DataFrame): Los datos cargados desde el CSV.
        model_filepath (str): La ruta del archivo del modelo de Regresión Logística (.pkl).

    Returns:
        str: La explicación generada por la IA sobre la predicción, o un mensaje de error.
    """
    
    # 1. Validar la entrada del DataFrame
    if df is None or df.empty:
        return "Error: El DataFrame está vacío o no se ha cargado correctamente."

    # 2. Cargar el Modelo de Regresión Logística
    try:
        modelo = joblib.load(model_filepath)
        # Intentar obtener los nombres de las características para asegurar la consistencia
        feature_names = list(modelo.feature_names_in_) if hasattr(modelo, 'feature_names_in_') else None
    except FileNotFoundError:
        return f"Error: No se encontró el archivo del modelo en la ruta: {model_filepath}"
    except Exception as e:
        return f"Error al cargar el modelo: {e}"

    # 3. Preprocesar los Datos
    try:
        # Tomar la primera fila para la predicción, asumiendo que el usuario quiere predecir una muestra
        datos_prediccion = df.iloc[0].to_frame().T 

        # Asegurar que las columnas del DataFrame coincidan con las del modelo
        if feature_names and not all(col in datos_prediccion.columns for col in feature_names):
             missing_cols = [col for col in feature_names if col not in datos_prediccion.columns]
             return f"Error: Faltan las siguientes columnas en el CSV: {', '.join(missing_cols)}. Asegúrate de que el CSV tenga las características correctas."
        
        # Filtrar solo las características que necesita el modelo
        if feature_names:
             datos_prediccion = datos_prediccion[feature_names]

    except Exception as e:
        return f"Error al preprocesar el DataFrame: {e}"

    # 4. Hacer la Predicción
    try:
        # Predicción de clase (0 o 1)
        prediccion_clase = modelo.predict(datos_prediccion)[0]
        # Probabilidades para un prompt más informativo
        prediccion_proba = modelo.predict_proba(datos_prediccion)[0]
        
        clase_predicha = "alta" if prediccion_clase == 1 else "baja"
        
        # Convertir datos del primer registro a un string legible para el prompt
        datos_string = datos_prediccion.iloc[0].to_dict()
        datos_str_limpio = ", ".join([f"{k}: {v:.2f}" for k, v in datos_string.items()])
        
    except Exception as e:
        return f"Error durante la predicción con el modelo: {e}"

    # 5. Generar la Explicación Textual con la IA
    try:
        prompt_explicacion = (
            f"Actúa como un analista de datos. El modelo de clasificación predijo **{clase_predicha}** calidad. "
            f"Los datos de entrada fueron: {datos_str_limpio}. "
            f"La probabilidad de la clase 'alta' es: {prediccion_proba[1]:.2f}. "
            f"Genera una explicación concisa y legible, mencionando las características clave (como acidez y azúcar) "
            f"que probablemente influyeron en el resultado. Usa un formato como: 'Este producto tiene alta/baja calidad porque...'"
        )

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt_explicacion}],
            temperature=0.3,
            max_tokens=200
        )
        explicacion = response.choices[0].message.content.strip()

        return explicacion

    except Exception as e:
        return f"Error al generar la explicación con la IA: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# App traductor")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Texto", lines=5)
            lang = gr.Dropdown(choices=LANGUAGES, label= "Elige un lenguaje",value="English")
            translate = gr.Button("Traducir")
            output = gr.Textbox(label="Traducción", lines=5)
            translate.click(fn=traductor, inputs=[text, lang], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)