import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import io
import os
import sys
import subprocess
import time
import pickle
import shutil

from io import StringIO

from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm.openai import OpenAI

from PIL import Image


OPENAI_API_KEY = ""

# Set Streamlit layout
st.set_page_config(layout="wide")


def chat_with_input_data(df, prompt, temperature, top_p):
    llm = OpenAI(api_token=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=temperature, top_p=top_p)

    # Configure SmartDataframe to use our custom plotting function
    config = {
        "llm": llm,
        "enable_cache": False,
        "verbose": False,
    }

    if os.path.isfile('/app/exports/charts/temp_chart.png'):
        os.remove('/app/exports/charts/temp_chart.png')

    sdf = SmartDataframe(df, config=config)
    result = sdf.chat(prompt)

    #print(sdf.last_code_generated)

    info = sdf.last_code_generated.split("\n")[-1].split("=")[1]
    #print(info)    

    if "dataframe" in info:
        info = "dataframe"
    elif "plot" in info:
        info = "plot"
    else:
        info = "string"
    
    return result, info

def preprocessing_page(temperature = 0.0, top_p = 0.1):
    st.header("Data Exploration and Preprocessing🕵️‍♀️", divider='rainbow')

    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    if input_csv is not None:

            col1, col2 = st.columns([1,1])

            with col1:
                st.info("Visualize CSV:")
                data = pd.read_csv(input_csv, header=0)
                st.dataframe(data, use_container_width=True)
                st.success(", ".join([x[0] for x in os.walk(os.getcwd())]))


            with col2:
                st.info("Interact with data:")
                input_text = st.text_area(f"Enter your query below, temperature: {temperature} and top_p: {top_p}")
                MAX_RETRIES = 5
                RETRY_DELAY = 3
                assumption = f"""The input dataframe has {data.shape[0]} rows and {data.shape[1]} columns.Always return results for the 
                            entire dataframe unless specifically asked\n

                            Assume the dataframe has some of the following columns with corresponding descriptions and context:\n
                             | Column Name | Column Units | Description Column | Group |
                            | ------------|--------------|---------------------|-------|
                            |SurfaceHoleLongitude|Decimal Degrees| The Longitude of the surface hole location|Location|
                            |SurfaceHoleLatitude|Decimal Degrees| The Latitude of the surface hole location|Location|
                            |BottomHoleLongitude|Decimal Degrees| The Longitude of the bottom hole location|Location|
                            |BottomHoleLatitude|Decimal Degrees| The Latitude of the bottom hole location|Location|
                            | Operator | None (string) | Company that operates the well | Completion |
                            |CompletionDate | None (date) |Date in which the well was completed| Completion|
                            |Reservoir | None (string) | Geologic formation that the well is targeting |Geology|
                            |LateralLength_FT | Feet | Completed length of the horizontal well |Completion|
                            |ProppantIntensity_LBSPerFT | Pounds / Feet | Amount of proppant (frac sand) per lateral foot used to complete the well | Completion
                            |FluidIntensity_BBLPerFT | Barrels / Feet | Amount of fluid per lateral foot used to complete the well | Completion
                            |HzDistanceToNearestOffsetAtDrill | Feet | Horizontal distance to the nearest offset well - measured at the time the well was completed | Well spacing
                            |HzDistanceToNearestOffsetCurrent | Feet | Horizontal distance to the nearest offset well - measured at current time | Well spacing
                            |VtDistanceToNearestOffsetCurrent | Feet | Vertical distance to the nearest offset well - measured at the time the well was completed | Well spacing
                            |VtDistanceToNearestOffsetAtDrill | Feet | Vertical distance to the nearest offset well - measured at current time | Well spacing
                            |WellDepth                       | Feet | Depth of the horizontal well | Geology
                            |ReservoirThickness | Feet | Thickness of the targeted reservoir | Geology
                            |OilInPlace | Million barrels of oil / square mile | Amount of oil in place for the target reservoir | Geology
                            |Porosity | Percent | Porosity of the target reservoir | Geology
                            |ReservoirPressure | PSI |Pressure of the target reservoir | Geology
                            |WaterSaturation | Percent | % saturation of water in the target reservoir fluid | Geology
                            |StructureDerivative | Percent | % change in depth of the target formation - proxy for geologic faults | Geology
                            |TotalOrganicCarbon | Percent | % of total organic carbon of the target formation | Geology
                            |ClayVolume | Percent | % clay of the target reservoir | Geology
                            |CarbonateVolume | Percent | % carbonate of the target reservoir | Geology
                            |Maturity |Percent |Maturity of the target reservoir |Geology|
                            |TotalWellCost_USDMM |Millions of dollars |Total cost of the horizontal well |Completion|
                            |CumOil12Month |Barrels of oil| Amount of oil produced in the first 12 months of production |Production
                            |rowID |None (ID) |unique identifier for each well| ID
                            \n
                             """
                if st.button("Chat with CSV") and input_text:
                    with st.spinner("Processing..."):
                        for attempt in range(MAX_RETRIES):
                            try:
                                result, info = chat_with_input_data(data, assumption+input_text, temperature, top_p)
                                break  # If successful, break out of the retry loop
                            except Exception as e:
                                if attempt < MAX_RETRIES - 1:  # If not the last attempt
                                    time.sleep(RETRY_DELAY)
                                else:
                                    st.error("No response found!")
                                    result, info = None, None
                        
                        if info == "plot":
                            # if os.path.isfile('temp_chart.png'):
                            #     print("Yes")
                            #     im = plt.imread('temp_chart.png')
                            #     st.image(im)
                            #     os.remove('temp_chart.png')
                            time.sleep(3)
                            st.image("/app/exports/charts/temp_chart.png")
                        elif info == "string":
                            if result is not None:
                                st.success(result)
                        else:
                            if result is not None:
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                else:
                                    st.dataframe(StringIO(result))
                else:
                    st.warning("Please enter a prompt")

    # Close any remaining figures
    plt.close('all')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_code_from_openai(prompt, temperature=0.0, top_p=0.1):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,  # Maximum for GPT-3.5 Turbo
            n=1,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def run_code(script_path, train_path, test_path, output_placeholder):
    try:  

        # Run the script with the paths to the temporary files
        process = subprocess.Popen(
            [sys.executable, script_path, train_path, test_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        output = []
        for line in process.stdout:
            output.append(line)
            # Update the placeholder with all output so far
            output_placeholder.text('\n'.join(output))
        
        # Wait for the process to complete and get the return code
        return_code = process.wait()
        
        # If there was an error, print the error output
        if return_code != 0:
            st.error("Error occurred. Error output:")
            for line in process.stderr:
                st.text(line)
            return False
        
        return True
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False


def training_page(temperature = 0.0, top_p = 0.1):
    st.header("Model Training⚙️", divider='rainbow')

    train_csv = st.file_uploader("Upload your Train CSV file", type=['csv'])
    test_csv = st.file_uploader("Upload your Test CSV file", type=['csv'])

    if train_csv and test_csv:

        train_data = pd.read_csv(train_csv, header=0)
        test_data =  pd.read_csv(test_csv, header=0)

        train_data.to_csv("train.csv", header=True, index=False)
        test_data.to_csv("test.csv", header=True, index=False)

        col1, col2 = st.columns([1,1])

        with col1:
            #st.subheader("Train and test data")
            st.info(f"Shape of train: {train_data.shape} and Shape of test: {test_data.shape}")

            st.text("Train summary:")
            st.dataframe(train_data.describe())

            st.divider()

            st.text("Test summary:")
            st.dataframe(test_data.describe())
        
        with col2:

            st.info("Interact with data:")
            input_text = st.text_area(f"Enter your request for a classification model below, \
                temperature: {temperature} and top_p: {top_p}")
            if st.button("Train model") and input_text:
                assumption = "Assume train and test data are passed to script as command line arguments and both contain the column CumOilCategory \
                which is the label to predict during classification. Always import os and sys libraries. Calculate all possible metrics for \
                classification task including classification report in scikit-learn library. Also, calculate feature importances with corresponding \
                feature names if possible. Add numbered print statements detailing progress throughout the script. Always print precision, recall, and F1 score at the end. \
                Return all the calculated metrics for the model as a dictionary where the values are of type dataframe. Save the dictionary as a pickle file called results.pkl.\
                Generate code for:"
                with st.spinner("Generating code..."):
                    result = get_code_from_openai(assumption+"\n"+input_text, temperature, top_p)
                    code_run = result.split("```python")[1].split("```")[0]
                    with st.expander("AI generated code"):
                        st.code(code_run, language="python")
                    file_path = f"generated_code.py"
                    with open(file_path, 'w') as file:
                        file.write(code_run)
                with st.spinner("Executing code..."):
                    output_placeholder = st.empty()
                    run_status = run_code(file_path, "train.csv", "test.csv", output_placeholder)
                    if run_status:
                        st.success("Training and prediction completed successfully!")

def chat_with_result_data(df, prompt, temperature, top_p):
    llm = OpenAI(api_token=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=temperature, top_p=top_p)

    # Configure SmartDataframe to use our custom plotting function
    config = {
        "llm": llm,
        "enable_cache": False,
        "verbose": False,
    }

    if os.path.isfile('/app/exports/charts/temp_chart.png'):
        os.remove('/app/exports/charts/temp_chart.png')

    sdf = SmartDataframe(df, config=config)
    result = sdf.chat(prompt)

    info = sdf.last_code_generated.split("\n")[-1].split("=")[1] 

    if "dataframe" in info:
        info = "dataframe"
    elif "plot" in info:
        info = "plot"
    else:
        info = "string"
    
    return result, info

def interpretation_page(temperature=0.0, top_p=0.1):
    st.header("Interpretation and Visualization📊", divider='rainbow')

    # Initialize session state for storing results
    if 'res_dict' not in st.session_state:
        st.session_state.res_dict = None

    # Option to choose between file upload and direct loading
    load_option = st.radio("Choose how to load results:", ("Upload File", "Load from Server"))

    if load_option == "Upload File":
        results = st.file_uploader("Upload your results pickle", type=['pkl'])

        if results is not None and st.session_state.res_dict is None:
            # Read and process the file only if it hasn't been processed before
            file_contents = results.read()
            file_object = io.BytesIO(file_contents)
            st.session_state.res_dict = pickle.load(file_object)
            st.success("Results loaded successfully from uploaded file.")

    else:  # Load from Server
        if st.button("Load results.pkl from server"):
            try:
                with open('/app/results.pkl', 'rb') as file:
                    st.session_state.res_dict = pickle.load(file)
                st.success("Results loaded successfully from server.")
            except FileNotFoundError:
                st.error("results.pkl not found on the server.")
            except Exception as e:
                st.error(f"An error occurred while loading the file: {str(e)}")

    if st.session_state.res_dict:
        st.success(f"Results report contains: {', '.join(list(st.session_state.res_dict.keys()))}.")
        
        # df_list = [st.session_state.res_dict[k] for k in st.session_state.res_dict]

        # Create a selectbox for choosing a specific dataframe
        selected_df_key = st.selectbox("Select a dataframe to chat with:", 
        list(st.session_state.res_dict.keys()), placeholder="Select contact method...")
        
        # Get the selected dataframe
        selected_df = st.session_state.res_dict[selected_df_key]

        # # Create a form for input and button
        # with st.form(key='query_form'):
        #     input_text = st.text_area(f"Enter your query, temperature: {temperature} and top_p: {top_p}")
        #     submit_button = st.form_submit_button(label='Chat with CSV')

        # Create a form for input and button
        with st.form(key='query_form'):
            input_text = st.text_area(f"Enter your query about '{selected_df_key}', temperature: {temperature} and top_p: {top_p}")
            submit_button = st.form_submit_button(label='Chat with CSV')

        if submit_button and input_text:
            with st.spinner("Processing..."):
                MAX_RETRIES = 5 
                RETRY_DELAY = 3
                assumption = "Answer questions for this dataframe only.\n\
                    Context: If the dataset is a classification report, The classification report dataframes \
                    contains columns 0, 1, 2 they refer to labels corresponding to 'Low', 'Medium', and 'High' Oil production classes and they can be used interchangably.\
                    Other columns may be present. The classification report dataframes also contains metrics Precision, Recall, F1-score and support for each label as its indices.\
                    Extract correponding values for each label and metric." #Support refers to the number of samples used to calculate the scores and is always an integer. If present, feature importances\
                    #dataframe contains features and their importances as float values. If asked, All plots should have tick labels and font sizes adjusted for clarity and saved.\n"
                
                for attempt in range(MAX_RETRIES):
                    try:
                        result, info = chat_with_result_data(selected_df, input_text, temperature, top_p)
                        break  # If successful, break out of the retry loop
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:  # If not the last attempt
                            time.sleep(RETRY_DELAY)
                        else:
                            st.error(f"No response found! Error: {str(e)}")
                            result, info = None, None

                if info == "plot":
                    # if os.path.isfile('temp_chart.png'):
                    #     print("Yes")
                    #     im = plt.imread('temp_chart.png')
                    #     st.image(im)
                    #     os.remove('temp_chart.png')
                    time.sleep(3)
                    st.image("/app/exports/charts/temp_chart.png")
                elif info == "string":
                    if result is not None:
                        st.success(result)
                else:
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.dataframe(StringIO(result))
        elif submit_button and not input_text:
            st.warning("Please enter a prompt")




# Main app
def main():
    # Create two columns
    col1, col2 = st.columns([1, 8])  # Adjust the ratio as needed

    # Column 1: Logo
    logo_path = "images/logo.jpg"  # Adjust this path to your logo file
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        col1.image(logo, width=100)  # Adjust width as needed
    else:
        col1.error("Logo not found")

    # Column 2: Title
    col2.title("Quantum Midland WeLLM App")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Preprocessing", "Training", "Interpretation"])


    # Add slider to the sidebar
    st.sidebar.header("LLM Settings")
    temperature = st.sidebar.select_slider(
        "Select a value for temperature:",
        options=[0, 0.1, 0.25, 0.5, 0.75, 1.0])
    
    top_p = st.sidebar.select_slider(
        "Select a value for Top p:",
        options=[0, 0.1, 0.25, 0.5, 0.75, 1.0])
    
    # Display current slider value
    st.sidebar.info(f"Temperature: {temperature}")
        # Display current slider value
    st.sidebar.info(f"Top p: {top_p}")
    

    if page == "Preprocessing":
        preprocessing_page(temperature, top_p)
    elif page == "Training":
        training_page(temperature, top_p)
    else:
        interpretation_page(temperature, top_p)

if __name__ == "__main__":
    #plt.ioff()  # Turn off interactive mode globally
    main()






