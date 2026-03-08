import plotly.express as px

def eda(data):
    """
    Function for making plots
    :arg : data / Pandas Data Frame
    :return : Plotly figures
    """

    # Amount of chickens Vs Total egg production
    fig_1 = px.scatter(
        data,
        x="Amount_of_chicken",
        y="Total_egg_production",
        marginal_x="histogram",
        marginal_y="histogram",
        trendline="ols",
        title="Total Egg Production Vs Amount of chicken",
        labels={
            "Amount_of_chicken": "Amount of Chickens",
            "Total_egg_production": "Total Egg Production"
        },
        color_discrete_sequence=["crimson"],
        template="plotly_white"
    )
    # Amount of Feeding Vs Total egg production
    fig_2 = px.scatter(
        data,
        x="Amount_of_Feeding",
        y="Total_egg_production",
        marginal_x="histogram",
        marginal_y="histogram",
        trendline="ols",
        title="Total Egg Production Vs Amount of Feeding",
        labels={
            "Amount_of_Feeding": "Amount of Feeding",
            "Total_egg_production": "Total Egg Production"
        },
        color_discrete_sequence=["crimson"],
        template="plotly_white"
    )
    # Amounts of Ammonia Vs Total egg production
    fig_3 = px.scatter(
        data,
        x="Ammonia",
        y="Total_egg_production",
        marginal_y="histogram",
        marginal_x="histogram",
        trendline="ols",
        title="Ammonia amount Vs Total Egg Production",
        labels={
            "Ammonia": "Amounts of Ammonia",
            "Total_egg_production": "Total Egg Production"
        },
        color_discrete_sequence=["crimson"],
        template="plotly_white"
    )
    # Temperature in C Vs Total egg production
    fig_4 = px.scatter(
        data,
        x="Temperature",
        y="Total_egg_production",
        marginal_x="histogram",
        marginal_y="histogram",
        trendline="ols",
        title="Temperature Vs Total Egg Production",
        labels={
            "Temperature": "Temperature (C)",
            "Total_egg_production": "Total Egg Production"
        },
        color_discrete_sequence=["crimson"],
        template="plotly_white"
    )
    # Correlation Heatmap
    cor = data.corr()
    fig_5 = px.imshow(
        cor,
        text_auto = True,
        color_continuous_scale = "RdBu",
        origin = "upper",
        aspect = "auto",
        title = "Correlation Heatmap"
    )
    # Return plots
    plots = {
        "chicken_amount_total_egg" : fig_1,
        "feeding_amount_total_egg" : fig_2,
        "ammonia_amount_total_egg" : fig_3,
        "temperature_c_total_egg"  : fig_4,
        "correlation_heat_map" : fig_5
    }
    return plots


