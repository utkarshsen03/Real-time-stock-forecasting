import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime
from googlesearch import search
from stocknews import StockNews


# Machine Learning and Advanced Analytics
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

def get_url(query):
    results = search(query,num_results=1,safe=None)
    if results:
        return results
    else:
        return ""

def calculate_risk_metrics(financial_data,symbol):
    """
    Advanced risk calculation module
    """
    # Check for NaN values and drop them
    if financial_data['Close'][symbol].isnull().any():
        financial_data = financial_data.dropna(subset=['Close'])

    returns = financial_data['Close'][symbol].pct_change().dropna()  # Drop NaN returns
    
    # Ensure returns are not empty
    if returns.empty:
        return {
            'value_at_risk': None,
            'conditional_var': None,
            'sharpe_ratio': None,
            'max_drawdown': None
        }

    risk_metrics = {
        'value_at_risk': np.percentile(returns, 5),  # 5% VaR
        'conditional_var': returns[returns <= np.percentile(returns, 5)].mean(),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(financial_data['Close'])
    }
    
    return risk_metrics

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio with adjustable risk-free rate
    """
    excess_returns = returns - (risk_free_rate / 252)  # Annualized risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices):
    """
    Calculate maximum portfolio drawdown
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())

def calculate_rsi(prices, periods=14):
    """
    Calculate Relative Strength Index (RSI)
    """
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def ml_enhanced_prediction(financial_data,label):
    """
    Machine Learning enhanced prediction model using XGBoost
    Updated to handle feature consistency and data validation
    """
    # Validate input data
    if financial_data is None or financial_data.empty:
        raise ValueError("Input financial data is empty")

    # Ensure we have the necessary columns
    required_columns = pd.Index(['Close', 'Volume'])  # Convert to Index
    if not all(col in financial_data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Ensure we have enough data points for feature engineering
    if len(financial_data) < 50:  # Need at least 50 points for MA_50
        raise ValueError("Insufficient data points. Need at least 50 data points for feature engineering")

    # Feature engineering with robust handling
    features = pd.DataFrame({
        'Close': financial_data['Close'][label],
        'MA_10': financial_data['Close'][label].rolling(window=10, min_periods=1).mean(),
        'MA_50': financial_data['Close'][label].rolling(window=50, min_periods=1).mean(),
        'RSI': calculate_rsi(financial_data['Close'][label]),
        'Volume': financial_data['Volume'][label].fillna(method='ffill')  # Forward fill any missing volume data
    })
    
    # Drop any remaining NaN values and validate
    features = features.dropna()
    if len(features) == 0:
        raise ValueError("No valid data points after feature engineering")
    
    # Prepare data
    X = features[['MA_10', 'MA_50', 'RSI', 'Volume']]
    y = features['Close']
    
    # Ensure we have enough data for training
    if len(X) < 10:  # Minimum threshold for meaningful training
        raise ValueError("Insufficient data points after preprocessing")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features with error handling
    try:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X)
        # X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        raise ValueError(f"Error during feature scaling: {str(e)}")
    
    # Train XGBoost model
    try:
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train_scaled, y)
    except Exception as e:
        raise ValueError(f"Error during model training: {str(e)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': xgb_model,
        'feature_importance': feature_importance,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(),
        'data_points': len(features)  # Add information about the number of valid data points
    }

def generate_forecast(model, scaler, last_data,symbol, forecast_period=30):
    """
    Generate future price forecasts using the trained model
    Updated to handle feature consistency and prevent scaling errors
    """
    # try:
        # Validate input data
    if last_data is None or len(last_data) < 50:
        raise ValueError("Insufficient data for forecasting. Need at least 50 data points.")
        
    # Prepare feature data with the exact columns used in training
    feature_data = pd.DataFrame({
        'MA_10': last_data['Close'][symbol].rolling(window=10, min_periods=1).mean(),
        'MA_50': last_data['Close'][symbol].rolling(window=50, min_periods=1).mean(),
        'RSI': calculate_rsi(last_data['Close'][symbol]),
        'Volume': last_data['Volume'][symbol].fillna(method='ffill')  # Forward fill missing values
    })
    
    # Drop any remaining NaN values
    feature_data = feature_data.dropna()
    if len(feature_data) == 0:
        raise ValueError("No valid data points after feature engineering")
        
    # Get the last complete row of features
    last_features = feature_data.iloc[-1:]
    
    if last_features.empty:
        raise ValueError("Could not get valid features for prediction")
        
    
    # Generate predictions
    predictions = []
    current_features = last_features.copy()
    
    for _ in range(forecast_period):
        # Make prediction
        #try:
            current_scaled = scaler.transform(current_features)
            pred = model.predict(current_scaled)
            predictions.append(pred[0])
            
            # Update features for next prediction (simplified)
            new_features = pd.DataFrame({
                'MA_10': [pred[0]],
                'MA_50': current_features['MA_50'].iloc[-1],
                'RSI': current_features['RSI'].iloc[-1],
                'Volume': current_features['Volume'].iloc[-1]
            })
            current_features = new_features
            
        #except Exception as e:
            #raise ValueError(f"Error during prediction iteration:{e}")
    
    # Create forecast dates
    forecast_dates = pd.date_range(
        start=last_data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_period,
        freq='B'  # Business days
    )
    
    return pd.Series(predictions, index=forecast_dates)

def create_advanced_forecast_plot(historical_data, forecast_data, symbol, ml_model=None):
    """
    Create an advanced interactive plot with multiple data layers
    """
    # Prepare historical data
    historical_df = pd.DataFrame(historical_data)
    
    # Rename columns to handle different data sources
    historical_columns = list(historical_df.columns)
    if 'Close' not in historical_columns and len(historical_columns) > 0:
        historical_df.columns = ['Close'] + historical_columns[1:]
    
    # Create figure with multiple traces
    title = f"Advanced {symbol} Forecast"
    fig = px.line(title=title)
    
    # Historical price
    fig.add_scatter(x=historical_df.index, y=historical_df['Close'], 
                    name='Historical Price', line=dict(color='blue'))
    
    # Forecast trace
    forecast_values = forecast_data.iloc[:, 0]  # Take first column
    fig.add_scatter(x=forecast_data.index, y=forecast_values,
                    name='Forecast', line=dict(dash='dash', color='red'))
    
    # Optional ML prediction confidence interval
    if ml_model is not None:
        # Placeholder for ML model confidence interval
        pass
    
    # Customize layout with advanced options
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Price Trend',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig



def main():
    st.title("Stock Dashboard")
    with st.sidebar:
        st.header("Analysis Configuration")
        tickers = pd.read_csv('tickers.csv')
        symbols = list(tickers["symbol"])
        analysis = st.radio("",["Single Ticker","Compare Stocks"])
        if analysis == "Single Ticker":
            ticker = st.selectbox("Select Ticker",symbols)
        elif analysis == "Compare Stocks":
            ticker = st.selectbox("Select Ticker",symbols)
            symbols.remove(ticker)
            ticker2 = st.selectbox("Select Second Ticker",symbols)
        period = st.sidebar.selectbox("Analysis Timeframe",["6mo", "1y", "2y", "5y"],index=1,
                                      help="Select historical data timeframe for analysis")
        if period == "6mo":
            start_date = datetime.date.today() - datetime.timedelta(days=180)
        elif period == "1y":
            start_date = datetime.date.today() - datetime.timedelta(days=365)
        elif  period == "2y":
            start_date = datetime.date.today() - datetime.timedelta(days=730)
        else:
            start_date = datetime.date.today() - datetime.timedelta(days=1825)
        end_date = datetime.date.today()

    stock = yf.Ticker(ticker)
    info = stock.info
    left, middle, right = st.columns([2,1,2])
    try:
        left.write(f"**Company:** {info['longName']}")
        middle.write(f"**Sector:** {info['sector']}")
        right.write(f"**Industry:** {info['industry']}")
        left.write(f"**Market Cap:** {info['marketCap']}")
        middle.write(f"**P/E Ratio:** {info['trailingPE']}")
    except Exception as e:
        pass


    pricing_data,dashboard,forecasting, news = st.tabs(["Pricing Data","Dashboard","Forecasting","Top 10 News"])

    data = yf.download(ticker,period=period, 
            interval='1d',  # Daily data
            progress=False)

    with pricing_data:
        if len(data) < 2:
            st.error("Insufficient data for analysis.")
        else:
            risk_metrics = calculate_risk_metrics(data, ticker)
        st.subheader("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk-Adjusted Return (Sharpe)", f"{risk_metrics['sharpe_ratio']:.2f}")
        with col2:
            st.metric("Value at Risk (95%)", f"{risk_metrics['value_at_risk']:.2%}")
        with col3:
            st.metric("Maximum Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
            
        st.header("Recent Market Data")
        new_data2 = data.copy()
        new_data2["Return"] = (data["Close"] / data["Close"].shift(1)) - 1
        new_data2['Cummulative Return'] = (1+new_data2['Return']).cumprod()
        new_data2.dropna(inplace=True)
        st.write(data)
        annual_return = new_data2["Return"].mean()*252*100
        st.info(f"Annual Return in Price: {annual_return}%")
        st.info(f"If a person has invested 1 unit of local currency in {start_date}, it would have been {round(new_data2['Cummulative Return'].iloc[-1],2)} by {end_date}")

    with dashboard:
        # try:
            new_data = data["Adj Close"].reset_index().melt(id_vars="Date",var_name="Close",value_name="value")
            fig = px.line(new_data,x="Date",y="value",title=f"{info['longName']} Closing Price")
            st.plotly_chart(fig)
            
            st.subheader(f"Change in Return of {ticker}")
            st.bar_chart(new_data2["Return"])
            
            st.subheader(f"Cummulative Return of {ticker}")
            st.line_chart(new_data2["Cummulative Return"])
            
            MA_value1 = st.slider("Select First Moving Average",0,1000,value=None)
            MA_value2 = st.slider("Select Second Moving Average",0,1000,value=None)
            new_data['1_MA'] = new_data["value"].rolling(MA_value1).mean()
            new_data["2_MA"] = new_data["value"].rolling(MA_value2).mean()
            fig = px.line(new_data,x="Date",y=["value","1_MA","2_MA"],title=f"{info['longName']}{MA_value1} and {MA_value2} Moving Average")
            fig.for_each_trace(lambda trace: trace.update(name=trace.name.replace("value","Closing Price").replace(
                "1_MA","First Moving Average Price").replace("2_MA","Second Moving Average Price")))
            st.plotly_chart(fig)
            
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                        open=data.Open[ticker],
                        high=data.High[ticker],
                        low=data.Low[ticker],
                        close=data.Close[ticker]))
            fig.update_layout(title=f'Candlestick Chart for {ticker}',
                        xaxis_title='Date',
                        yaxis_title='Price')
            st.plotly_chart(fig)
            
            
            
            if analysis == "Compare Stocks":
                ticker2_data = yf.download(ticker2,start=start_date,end=end_date)
                ticker2_new_data = ticker2_data["Adj Close"].reset_index().melt(id_vars="Date",var_name="Close",value_name="value")
                df = pd.merge(data,ticker2_data,on="Date")
                fig = px.line(x=df.index,y=[df["Close"][ticker],df["Close"][ticker2]],markers=True,title=f"{ticker} and {ticker2} Closing Price")
                fig.for_each_trace(lambda trace: trace.update(name=trace.name.replace("wide_variable_0",ticker).replace("wide_variable_1",ticker2)))
                st.plotly_chart(fig)
                
                
                st.subheader("Scatter Graph")
                column_1 = st.selectbox(f"Column of{ticker}:",data.columns,index=None)
                column_2 = st.selectbox(f"Column of {ticker2}:",ticker2_data.columns,index=None)
                
                if column_1 != None and column_2 != None:
                    x_value = np.array(df[column_1])
                    y_value = np.array(df[column_2])
                    fig = px.scatter(x_value,y_value,title=f"{ticker} V/S {ticker2}",trendline="ols")
                    st.plotly_chart(fig)
                    
                
        # except Exception as e:
            # st.error("Invalid Ticker")

    with  forecasting:
        #try:
            # Ensure we have enough data
            if len(data) < 50:
                st.warning("Insufficient data for quantitative analysis. Need at least 50 data points.")
                return
                
            # Perform ML prediction with error handling
            ml_results = ml_enhanced_prediction(data,ticker)
            
            if ml_results and 'model' in ml_results:
                                
                with st.expander("Feature Analysis"):
                    st.write("### Feature Importance Analysis")
                    
                    # Display feature importance with interpretation
                    importance_df = ml_results['feature_importance']
                    st.dataframe(importance_df)
                    
                    # Interpret top features
                    top_feature = importance_df.iloc[0]
                    st.write("#### Key Driver Analysis")
                    st.write(f"The most significant factor in price movement is **{top_feature['feature']}** "
                            f"with an importance score of {top_feature['importance']:.2%}")
                    
                    # Feature correlation analysis
                    st.write("#### Feature Correlations")
                    feature_data = data[['Close', 'Volume']].copy()
                    feature_data['MA_10'] = data['Close'].rolling(window=10).mean()
                    feature_data['MA_50'] = data['Close'].rolling(window=50).mean()
                    feature_data['RSI'] = calculate_rsi(data['Close'])
                    
                    correlation_matrix = feature_data.corr()
                    fig = px.imshow(correlation_matrix,labels=dict(color="Correlation"),
                                    color_continuous_scale="RdBu")
                    st.plotly_chart(fig, use_container_width=True)
                                    # Create tabs for different analysis sections
                
                
                with st.expander("Technical Indicators"):
                    st.write("### Technical Analysis")
                    
                    # RSI Analysis
                    rsi = calculate_rsi(data['Close'])
                    current_rsi = float(rsi.iloc[-1])
                    
                    # Moving Average Analysis
                    ma_10 = data['Close'].rolling(window=10).mean()
                    ma_50 = data['Close'].rolling(window=50).mean()
                    
                    # Current price vs MAs
                    current_price = data['Close'].iloc[-1]
                    price_vs_ma10 = float((current_price / ma_10.iloc[-1] - 1) * 100)
                    price_vs_ma50 = float((current_price / ma_50.iloc[-1] - 1) * 100)
                    
                    # Technical signals
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RSI (14)", f"{current_rsi:.1f}", 
                                    help="RSI > 70: Overbought, RSI < 30: Oversold")
                        st.metric("Price vs MA(10)", f"{price_vs_ma10:+.1f}%")
                    with col2:
                        st.metric("Volume Trend", 
                                    "Above Average" if data['Volume'][ticker].iloc[-1] > data['Volume'][ticker].mean() else "Below Average")
                        st.metric("Price vs MA(50)", f"{price_vs_ma50:+.1f}%")
                    
                    # Technical Analysis Summary
                    st.write("#### Technical Outlook")
                    signals = []
                    if current_rsi > 70:
                        signals.append("• RSI indicates **overbought** conditions")
                    elif current_rsi < 30:
                        signals.append("• RSI indicates **oversold** conditions")
                    
                    if price_vs_ma10 > 0 and price_vs_ma50 > 0:
                        signals.append("• Price is **above** both moving averages, indicating an uptrend")
                    elif price_vs_ma10 < 0 and price_vs_ma50 < 0:
                        signals.append("• Price is **below** both moving averages, indicating a downtrend")
                    
                    for signal in signals:
                        st.write(signal)
                
                with st.expander("Forecast Analysis"):
                    st.write("### Forecast Analysis")
                    
                    # Generate forecast using sufficient historical data
                    forecast_window = min(60, len(data))
                    forecast_values = generate_forecast(
                        ml_results['model'], 
                        ml_results['scaler'], 
                        data.tail(forecast_window),
                        symbol=ticker,
                        forecast_period=30
                    )
                    
                    if forecast_values is not None:
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame(forecast_values, columns=['Forecast'])
                        
                        # Calculate forecast metrics
                        forecast_change = float((forecast_df['Forecast'].iloc[-1] / current_price - 1) * 100)
                        forecast_volatility = float(forecast_df['Forecast'].std() / forecast_df['Forecast'].mean() * 100)
                        
                        # Display forecast metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("30-Day Forecast Change", f"{forecast_change:+.1f}%")
                            st.metric("Forecast Volatility", f"{forecast_volatility:.1f}%")
                        with col2:
                            st.metric("Data Points Used", f"{ml_results.get('data_points', 'N/A')}")
                            st.metric("Forecast Window", f"{forecast_window} days")
                        
                        # Forecast visualization
                        fig = create_advanced_forecast_plot(
                            data['Close'],
                            forecast_df,
                            ticker,
                            ml_results['model']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast interpretation
                        st.write("#### Forecast Interpretation")
                        if forecast_change > 0:
                            st.write(f"The model predicts a **positive trend** with an expected return of {forecast_change:+.1f}% "
                                    f"over the next 30 days, with a volatility of {forecast_volatility:.1f}%")
                        else:
                            st.write(f"The model predicts a **negative trend** with an expected return of {forecast_change:+.1f}% "
                                    f"over the next 30 days, with a volatility of {forecast_volatility:.1f}%")
                
                with st.expander("Market Regime"):
                    st.write("### Market Regime Analysis")
                    
                    # Calculate returns
                    returns = data['Close'].pct_change()
                    volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
                    current_volatility = volatility.iloc[-1][0]
                    
                    
                    # Determine market regime
                    if current_volatility < 15:
                        regime = "Low Volatility"
                        regime_desc = "The market is currently in a stable state with low volatility"
                    elif current_volatility < 25:
                        regime = "Normal Volatility"
                        regime_desc = "The market is showing typical volatility levels"
                    else:
                        regime = "High Volatility"
                        regime_desc = "The market is experiencing elevated volatility levels"
                    
                    # Display regime analysis
                    st.metric("Current Market Regime", regime)
                    st.metric("Annualized Volatility", f"{current_volatility:.1f}%")
                    st.write(f"**Regime Description:** {regime_desc}")
                    
                    # Plot volatility trend
                    fig = px.line(volatility, 
                                title="Historical Volatility Trend",
                                labels={"value": "Annualized Volatility (%)", "index": "Date"})
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.warning("Could not perform quantitative analysis. Check the data quality.")
        #except Exception as e:
        #    st.error(f"Error in quantitative analysis: {str(e)}")


    with news:
        sn = StockNews(ticker,save_news=False)
        df_news = sn.read_rss()
        for i in range(10):
            url = get_url(df_news['title'][i])
            st.subheader(df_news['title'][i])
            st.write(url)
            st.write(df_news['published'][i])
            st.write(df_news['summary'][i])
            
if __name__ == "__main__":    
    main()
