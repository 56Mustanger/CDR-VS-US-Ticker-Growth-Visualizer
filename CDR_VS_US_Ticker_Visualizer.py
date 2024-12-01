
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import financedatabase as fd
import numpy as np
import statsmodels.api as sm



def has_header(file):
    try:
        sample = pd.read_csv(file, nrows=1)
        return all(isinstance(x, str) for x in sample.columns)
    except Exception as e:
        return False 
def returnc(df):
        rel = df.pct_change()
        cumret = ((1+rel).cumprod()-1)
        cumret = cumret.fillna(0)
        Delta['Delta'] = (cumret.iloc[:, 0] - cumret.iloc[:, 1])*100
        cumret = cumret*100
        return cumret

st.set_page_config(layout="wide")

st.title("CDRs :flag-ca: VS :flag-um: Ticker Growth Visualizer ")

tab1, tab2 = st.tabs(["Stock CDR Visualizer", "Custom Data"])
with tab1:
    col1, col2,col3,col4 = st.columns(4)
    st.subheader("Relative Growth Rate Between CDR vs US Dolar Equity")
    with col2:
        ticker = st.text_input("Enter Ticker","MSFT")
        ticker2 = ticker+".NE"
        ticker =[ticker,ticker2]

    with col3:
        fx1 = st.text_input('Intial % fee','0')
        fx2 = st.text_input('Final % fee','0')
        fx3 =float(fx1)+float(fx2)
        
    with col4:
        start_date = st.date_input("Start Date", value = pd.to_datetime('2022-11-09'))
        end_date = st.date_input("End Date", value = pd.to_datetime('today'))
    with col1:
        url = "https://cdr.cibc.com/#/cdrDirectory"
        st.subheader("How to Use:")
        st.write("1. Enter a ticker symbol that is traded in the US Dollar listed [here](%s) on a major exchange (NASDAQ/NYSE) and ensure the same ticker has an aviable CDR ticker on the NEO exchange. "  % url)
        st.write("2. Enter intial purchase fees / conversion rate and selling fees/conversion rate") 
        st.write("3. Enter chart visualization start and end dates You can input custom stock data with the \"custom\" tab above.")


    

    try:
        fx3 = float(fx1)+float(fx2)
        Delta ={'Delta':"",'Coversion Fees':(fx3)}
        df = yf.download(ticker, start=start_date, end=end_date)['Adj Close']

        e = returnc(df)
        st.line_chart(e)
        

        fig = px.line(Delta, labels={

                            "value": "% Growth",
                            "variable": "Delta Growth"
                        })
        fig.update_layout(
            title="",
            font_family="Arial",
            font_size = 30,
            font_color="red",
            title_subtitle_font_size =100,
            legend_font_size=10,
            hoverlabel_font_size = 15,
            newshape_legendgrouptitle_font_size =30
        )
        st.subheader("Relative profitability when accounting for fees and conversion rates")
        

        st.plotly_chart(fig)
        col1, col2 = st.columns(2)
        with col1:
            st.info("More profitable to Hold US Ticker untill the delta is greater than cumulative conversion fees.", icon="🇺🇸")
        with col2:
            st.error("More profitable to Hold CDR untill the delta is lower than cumulative conversion fees.", icon="🇨🇦")


        fig.update_layout(title_subtitle_font_size=50)
    except ValueError:
        st.warning("Please enter a numerical coversion fees")

    

with tab2:
    col7, col8,col9,col10, = st.columns(4)
    with col7:
        x_values = st.text_input(
            "Manual Enter X-Values seperated by commas","2,4,8,16")
        x_values = x_values.split(",")
        y_values = st.text_input(
                "Ticker Enter Y-Values seperated by commas","1,2,3,4")
        y_values = y_values.split(",")
        try:
            x_values = [float(i) for i in x_values]
            
            y_values = [float(i) for i in y_values]
        except ValueError:
            pass


    with col8:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df1=pd.read_csv(uploaded_file,header=None)
            x_values = df1[0].tolist() 
            y_values = df1[1].tolist()

        else:
            st.warning("If you have data locally, you can upload a basic csv file.(No headers, with the format of x and y columns)")




    user_fit = st.selectbox('Choose A fit', ['Linear','Lowess',"Logarithm","Expoential","Polynomial", "Area"])





    if uploaded_file is None:
        try:
            df1 = pd.DataFrame({'x': x_values, 'y': y_values})
        except KeyError:
            pass
    else:
         pass

    if user_fit =="Linear":
        fig = px.scatter(df1,x=x_values,y=y_values,title="Plotted Data", trendline="ols")
        st.plotly_chart(fig)
        results = px.get_trendline_results(fig)
        ols_results = results.iloc[0]["px_fit_results"] 


        x=pd.DataFrame({'coefficients' : ols_results.params,  
        'r_squared' : ols_results.rsquared,   
        'p_values' : ols_results.pvalues})

        c1= x['coefficients'].iloc[0]
        c2 = x['coefficients'].iloc[1]
        equation = f"{round(c1,2)}x+{round(c2,2)}"
        st.write("Linear Equation: f(x)=", equation) 
            
        st.table(x)
    
    if user_fit =="Lowess":
        fig = px.scatter(df1,x=x_values,y=y_values,title="Plotted Data", trendline="ols")
        st.plotly_chart(fig)

    if user_fit =="Area":
        fig = px.area(df1,x=x_values,y=y_values,title="Plotted Data")
        st.plotly_chart(fig)


    if user_fit =="Expoential":
        fig = px.scatter(df1,x=x_values,y=y_values,title="Plotted Data", trendline="ols",  trendline_options=dict(log_y=True))
        st.plotly_chart(fig)
        results = px.get_trendline_results(fig)
        ols_results = results.iloc[0]["px_fit_results"]  


 
        x=pd.DataFrame({'coefficients' : ols_results.params, 
        'r_squared' : ols_results.rsquared,  
        'p_values' : ols_results.pvalues}) 
            

        c1= x['coefficients'].iloc[0]
        c2 = x['coefficients'].iloc[1]
        equation = f"{round(c1,2)}^x+{round(c2,2)}"
        st.write("Exponetial Equation: f(x)=", equation) 

        st.table(x)

    if user_fit =="Logarithm":
        fig = px.scatter(df1,x=x_values,y=y_values,title="Plotted Data", trendline="ols",  trendline_options=dict(log_x=True))
        st.plotly_chart(fig)
        results = px.get_trendline_results(fig)
        ols_results = results.iloc[0]["px_fit_results"] 
      
        x=pd.DataFrame({'coefficients' : ols_results.params,
        'r_squared' : ols_results.rsquared,   
        'p_values' : ols_results.pvalues}) 

        c1= x['coefficients'].iloc[0]
        c2 = x['coefficients'].iloc[1]
        equation = f"log{round(c2,2)}(x)"
        st.write("Logarithmic Equation: f(x)=", equation) 
            
        st.table(x)


 

    if user_fit=="Polynomial":

        degree_input = st.text_input("Enter the degree of the polynomial:", value="2")
        degree = int(degree_input)

        if (0<degree):
            
            z = np.polyfit(x_values, y_values, degree)
            f = np.poly1d(z)

           
            x_new = np.linspace(min(x_values), max(x_values), 500)
            y_new = f(x_new)


            y_pred = f(x_values) 
            ss_res = np.sum((y_values - y_pred) ** 2)  
            ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)  
            r_squared = 1 - (ss_res / ss_tot)


            X = np.vander(x_values, degree + 1)  
            model = sm.OLS(y_values, X).fit()
            p_values = model.pvalues


            equation_terms = [
                f"{coef:.2f}X^{i}" for i, coef in enumerate(z[::-1])
            ]
            equation = " + ".join(equation_terms)

            trace1 = go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                marker=dict(color="rgb(255, 127, 14)", size=10),
                name="Data",
            )

            trace2 = go.Scatter(
                x=x_new,
                y=y_new,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=2),
                name="Polynomial Fit",
            )

            layout = go.Layout(
                title="Plotted Data",
                xaxis=dict(title="X-axis"),
                yaxis=dict(title="Y-axis"),
                showlegend=True,
            )

            fig = go.Figure(data=[trace1, trace2], layout=layout)

            

            st.plotly_chart(fig)


            coef_table = pd.DataFrame({
                "Coefficients": z[::-1],
                "P-values": list(p_values),
                "R-values":(r_squared)
            })

            st.write("Polynomial Equation f(x) =", equation)
            st.table(coef_table)
        else:
            st.error("Invalid input. Please enter an integer greater than or equal to 2")

        
