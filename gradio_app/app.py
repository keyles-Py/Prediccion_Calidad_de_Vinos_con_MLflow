from openai import OpenAI
import gradio as gr
import mlflow
import pandas as pd
import joblib

gemini_api_key = "AIzaSyDSJRhRW_0VlPpukX3Qlqbjz3YMC8gu0z8"

client = OpenAI(
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
  api_key=gemini_api_key,
)

def explicar_prediccion(
    csv_filepath: str, 
    model_filepath: str = "modelo_logreg.pkl"
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
    
    # 1. Cargar el Modelo de Regresión Logística
    try:
        modelo = joblib.load(model_filepath)
        feature_names = list(modelo.feature_names_in_) if hasattr(modelo, 'feature_names_in_') else None
    except FileNotFoundError:
        return f"Error: No se encontró el archivo del modelo en la ruta: {model_filepath}"
    except Exception as e:
        return f"Error al cargar el modelo: {e}"

    # 2. Leer el CSV con la ruta temporal
    try:
        if csv_filepath is None:
             return "Error: No se ha subido ningún archivo CSV."
             
        # Leer el archivo usando la ruta proporcionada por Gradio
        df = pd.read_csv(csv_filepath)
        
        if df.empty:
            return "Error: El archivo CSV está vacío."

        # Tomar la primera fila para la predicción
        datos_prediccion = df.iloc[0].to_frame().T 

        # Asegurar que las columnas coincidan
        if feature_names:
            missing_cols = [col for col in feature_names if col not in datos_prediccion.columns]
            if missing_cols:
                 return f"Error: Faltan las siguientes columnas en el CSV: {', '.join(missing_cols)}."
            datos_prediccion = datos_prediccion[feature_names]

    except FileNotFoundError:
        # Aunque Gradio proporciona una ruta, este error podría ocurrir si la ruta es inválida
        return f"Error al acceder al archivo temporal: {csv_filepath}"
    except Exception as e:
        return f"Error al leer o preprocesar el CSV: {e}"

    # 3. Hacer la Predicción
    try:
        prediccion_clase = modelo.predict(datos_prediccion)[0]
        prediccion_proba = modelo.predict_proba(datos_prediccion)[0]
        
        clase_predicha = "alta" if prediccion_clase == 1 else "baja"
        
        # Preparar los datos de entrada para la IA
        datos_string = datos_prediccion.iloc[0].to_dict()
        datos_str_limpio = ", ".join([f"{k}: {v:.2f}" for k, v in datos_string.items()])
        
    except Exception as e:
        return f"Error durante la predicción con el modelo: {e}"

    # 4. Generar la Explicación Textual con la IA
    try:
        prompt_explicacion = (
            f"Actúa como un analista de datos. El modelo de clasificación predijo **{clase_predicha}** calidad. "
            f"Los datos de entrada fueron: {datos_str_limpio}. "
            f"La probabilidad de la clase 'alta' es: {prediccion_proba[1]:.2f}. "
            f"Genera una explicación concisa y legible, mencionando las características clave (como acidez y azúcar) "
            f"que probablemente influyeron. Usa un formato como: 'Este producto tiene alta/baja calidad porque...'"
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

iface = gr.Interface(
    fn=explicar_prediccion,
    # 💡 Usa type="filepath" para recibir la ruta del archivo temporal
    inputs=gr.File(label="Sube el archivo CSV", type="filepath"), 
    outputs="text",
    title="Predicción de Calidad de Vino con Explicación IA"
)

iface.launch()