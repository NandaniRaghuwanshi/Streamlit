import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

train_data = pd.read_excel(r"D://ResoluteAi_Assignment//train.xlsx")

st.title("ML with Python task")
nav = st.sidebar.radio('Menu',['Task_1','Task_2','Task_3'])




if nav == 'Task_1':
    st.header('Task 1: Machine Learning - Clustering')
    if st.checkbox('Show target values'):
        target_values1 = pd.read_excel("D://ResoluteAi_Assignment//train.xlsx", nrows = 50)
        st.table(target_values1)
    
    shapeTask1 = train_data.shape
    st.write(f"☆ Shape of Training dataset: {shapeTask1}")
    
    st.write("☆ Clustering Algorithm: K-means Clustering")
    
    def predict_cluster(input1_data, model_clustering):
        cluster_pred = model_clustering.predict(input1_data)
        return cluster_pred

    def main():
        st.title("Clustering Prediction:")
    
        input1_data = collect_user_input1()
        if st.button("PREDICT"):
            cluster_prediction = predict_cluster(input1_data, model_clustering)
            st.success(f"The predicted cluster is: {cluster_prediction[0]}")
            
            silhouette_avg = silhouette_score(x1_train, train1_clusters)
            st.write(f"☆ Silhouette Score: {silhouette_avg}")
            st.write("The silhouette score, measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).")
            
            fig, ax = plt.subplots()
            scatter = ax.scatter(x1_train['T1'], x1_train['T2'], c = train1_clusters, cmap = 'viridis')
            ax.set_title('Training-Data Clusters')
            ax.set_xlabel('T1')
            ax.set_ylabel('T2')
            legend = ax.legend(*scatter.legend_elements(), title = 'Clusters')
            ax.add_artist(legend)
            st.pyplot(fig)
            
    def collect_user_input1():
        user_input1 = {}

        st.header("Enter Input:")
        for i in range(1, 19):
            user_input1[f'T{i}'] = st.number_input(f"Enter T{i}:")

        input1_series = pd.Series(user_input1)
        input1_df = pd.DataFrame([input1_series])
        return input1_df

    if __name__ == "__main__":
        x1_train = train_data.drop(columns=['target'])
        
        k = 3
        model_clustering = KMeans(n_clusters=k, random_state=1000)
        
        train1_clusters = model_clustering.fit_predict(x1_train)

        main()
        
        

if nav == 'Task_2':
    st.header('Task 2: Machine Learning - Classification')
    if st.checkbox('Show target values'):
        target_values2 = pd.read_excel("D://ResoluteAi_Assignment//train.xlsx", nrows = 50)
        st.table(target_values2)
    
    shapeTask2 = train_data.shape
    st.write(f"☆ Shape of Training dataset: {shapeTask2}")  
    
    st.write("☆ Classification Algorithm: RandomForestClassifier")
    st.write("☆ Following are some reasons behind choosing RandomForestClassifier algorithm for classification:\n"+
             "\n    - High Accuracy: Random Forests generally provide high accuracy compared to many other algorithms. They are effective in capturing complex relationships in the data, making them suitable for tasks where accuracy is crucial.\n"+
             "\n    - Robust to Overfitting: Random Forests are less prone to overfitting, especially when compared to decision trees. By combining multiple trees and aggregating their predictions, they can generalize well to new, unseen data.\n"+
             "\n    - Stability and Consistency: Random Forests are stable and consistent. They tend to provide reliable results across different runs and are less sensitive to the specific configuration of hyperparameters.\n"+
             "\n    - Ease of Implementation: Random Forests are relatively easy to implement and less sensitive to hyperparameter tuning compared to some other algorithms. This makes them a practical choice for practitioners without extensive machine learning expertise.")
    
    model_classification = RandomForestClassifier()
    
    y2_train = train_data['target']
    x2_train = train_data.drop(columns=['target'])
    
    x2_trg, x2_test, y2_trg, y2_test = train_test_split(x2_train, y2_train , test_size = 0.2 , random_state = 1000)

    model_classification.fit(x2_trg, y2_trg)

    y2_pred = model_classification.predict(x2_test)

    accuracyTask2 = accuracy_score(y2_pred, y2_test)
    st.write(f"☆ Accuracy Score: {accuracyTask2}")

    def predict2_output(input2_data):
        y2_test_pred = model_classification.predict(input2_data)
        return y2_test_pred

    def main():
        st.title("Target Prediction:")
        
        input2_data = collect_user_input2()
        if st.button("PREDICT"):
            prediction2 = predict2_output(input2_data)
            st.success(f"The predicted target is: {prediction2}")

    def collect_user_input2():
        user_input2 = {}

        st.header("Enter Input:")
        for i in range(1, 19):
            user_input2[f"T{i}"] = st.number_input(f"Enter T{i}:")

        input2_series = pd.Series(user_input2)
        input2_df = pd.DataFrame([input2_series])
        return input2_df

    if __name__ == "__main__":
        main()




if nav == 'Task_3':
    st.header('Task 3: Python')
    input_data = 'inputsheet'
    task3_data = pd.read_excel("D://ResoluteAi_Assignment//rawdata.xlsx",sheet_name = input_data  )
    if st.checkbox('Show raw data'):
        st.table(task3_data)
    shapeTask3 = task3_data.shape
    st.write(f"☆ Shape of Raw dataset: {shapeTask3}")  
    
    output_sheet = 'new_output'
    output_data = pd.read_excel("D://ResoluteAi_Assignment//rawdata.xlsx" , sheet_name = output_sheet)
    st.write(f"☆ Output :")
    st.write(output_data)
    