"""
Custom time series plotting functions implemented in plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pmdarima import ARIMA

RED_COLOR_DICT = {
    "95% PI": "rgba(255, 0, 0, 0.15)",  # Lightest red
    "90% PI": "rgba(255, 0, 0, 0.18)",
    "80% PI": "rgba(255, 0, 0, 0.21)",
    "70% PI": "rgba(255, 0, 0, 0.24)",
    "60% PI": "rgba(255, 0, 0, 0.27)",
    "50% PI": "rgba(255, 0, 0, 0.30)",
    "40% PI": "rgba(255, 0, 0, 0.33)",
    "30% PI": "rgba(255, 0, 0, 0.36)",
    "20% PI": "rgba(255, 0, 0, 0.39)",
    "10% PI": "rgba(255, 0, 0, 0.42)",  # Darkest red
    "forecast": "#660033",  # Darker yellow or deep gray
}


BLUE_COLOR_DICT = {
    "95% PI": "rgba(0, 0, 255, 0.15)",  # Lightest blue
    "90% PI": "rgba(0, 0, 255, 0.18)",
    "80% PI": "rgba(0, 0, 255, 0.21)",
    "70% PI": "rgba(0, 0, 255, 0.24)",
    "60% PI": "rgba(0, 0, 255, 0.27)",
    "50% PI": "rgba(0, 0, 255, 0.30)",
    "40% PI": "rgba(0, 0, 255, 0.33)",
    "30% PI": "rgba(0, 0, 255, 0.36)",
    "20% PI": "rgba(0, 0, 255, 0.39)",
    "10% PI": "rgba(0, 0, 255, 0.42)",  # Darkest blue
}


GREEN_COLOR_DICT = {
    "95% PI": "rgba(0, 128, 0, 0.15)",  # Lightest green
    "90% PI": "rgba(0, 128, 0, 0.18)",
    "80% PI": "rgba(0, 128, 0, 0.21)",
    "70% PI": "rgba(0, 128, 0, 0.24)",
    "60% PI": "rgba(0, 128, 0, 0.27)",
    "50% PI": "rgba(0, 128, 0, 0.30)",
    "40% PI": "rgba(0, 128, 0, 0.33)",
    "30% PI": "rgba(0, 128, 0, 0.36)",
    "20% PI": "rgba(0, 128, 0, 0.39)",
    "10% PI": "rgba(0, 128, 0, 0.42)",  # Darkest green
    "forecast": "#660033",  # Deep purple or dark gray
}

def generate_arima_prediction_intervals(
    model: ARIMA, 
    n_periods: int, 
    alpha: tuple = (0.05, 0.2)
) -> dict[str, pd.DataFrame]:
    """
    For a given ARIMA model generate and return the specified 
    prediction intervals. 

    Parameters:
    -----------
    model: ARIMA
        The trained ARIMA model. Note if the model is not trained the
        prediction will fail.

    n_periods: int
        The forecast horizon

    alpha: tuple, optional (default=(0.05, 0.2))
        Specified the 100(1-alpha)% prediction intervals to generate

    Returns
    -------
    dict{str: pd.DataFrame}
        A dictionary containing the prediction intervals. Each set has its own 
        key e.g. '95%' and '80%'.  The prediction intervals are a dataframe
        with a datetimeindex and two columns (lower, upper).
    
    """
    intervals_dict = {}
    
    for current_alpha in alpha:
        # forecast and prediction intervals
        preds, intervals = model.predict(
            n_periods=n_periods, return_conf_int=True, alpha=current_alpha
        )

        # cast to dataframe
        intervals = pd.DataFrame(
            intervals, index=preds.index, columns=["lower", "upper"]
        )

        # store in dict
        interval_str = f"{round((1 - current_alpha) * 100, 0)}%"
        intervals_dict[interval_str] = intervals

    return intervals_dict

def plot_time_series(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    prediction_intervals: dict[str, pd.DataFrame] | None = None,
    test_data_mode: str = "markers",
    y_axis_label: str = "Value",
    color_dict: dict | None = None,
    include_title=True
) -> None:
    """
    Plots univariate time series data using Plotly. Options for displaying
    training, test, and

    Parameters:
    ----------
    training_data: pd.DataFrame
        A pandas DataFrame with a DatetimeIndex containing the training data
        It should have one column representing the time series values.

    test_data: pd.DataFrame, optional (default=None)
        A pandas DataFrame with a DatetimeIndex containing the test data.
        Displayed as black dots or a line based on `test_data_mode`.

    forecast: pd.DataFrame, optional (default=None)
        A pandas DataFrame with a DatetimeIndex containing the forecasted data.

    prediction_intervals: dict[str, pd.DataFrame], Optional (default=None):
        A dictionary of pandas dataframes that contain prediction intervals
        Each dataframe contains two columns (upper and lower) and a 
        datetimeindex (over the forecast period).  The key for the dictionary
        can be set at your discertion. However, it is recommended to use
        convention '95% PI', '80% PI' etc.

    test_data_mode: str, optional (default='markers')
        Mode for displaying test data.
        Options are 'markers' (dots) or 'lines' (line).

    y_axis_label: str, optional (default='Value')
        The quantity measured by the time series to display on the y-axis

    color_dict: dict, optional (default=None)
        Dictionary with color codes for 'training', 'test', 'forecast',
        and prediction intervals.

    Returns:
        None: Displays an interactive Plotly figure.
    """
    # Validate test_data_mode input
    if test_data_mode not in ["markers", "lines"]:
        raise ValueError(
            "Invalid value for test_data_mode. Choose 'markers' or 'lines'."
        )

    if color_dict is None:
        color_dict = {}

    # Set default colors for standard elements
    color_dict.setdefault("training", "#0072B2")
    color_dict.setdefault("test", "#000000")
    color_dict.setdefault("forecast", "#FF0000")

    # Default prediction interval colors (red shades)
    default_pi_colors = [
        "rgba(255, 0, 0, 0.15)",  # Lightest red
        "rgba(255, 0, 0, 0.18)",
        "rgba(255, 0, 0, 0.21)",
        "rgba(255, 0, 0, 0.24)",
        "rgba(255, 0, 0, 0.27)",
        "rgba(255, 0, 0, 0.30)",
        "rgba(255, 0, 0, 0.33)",
        "rgba(255, 0, 0, 0.36)",
        "rgba(255, 0, 0, 0.39)",
        "rgba(255, 0, 0, 0.42)",  # Darkest red
    ]

    # Create a Plotly figure
    fig = go.Figure()

    # Add training data as a line plot
    fig.add_trace(
        go.Scatter(
            x=training_data.index,
            y=training_data.iloc[:, 0],
            mode="lines",
            name="Training Data",
            line=dict(color=color_dict["training"]),
        )
    )

    # Add prediction intervals before forecast
    if prediction_intervals is not None:
        for idx, (interval_name, interval_df) in enumerate(
            prediction_intervals.items()
        ):
            if not {"lower", "upper"}.issubset(interval_df.columns):
                err_msg = "DataFrame must contain 'lower' and 'upper' columns"
                raise ValueError(
                    f"{interval_name} {err_msg}"
                )

            interval_color = color_dict.get(
                interval_name, default_pi_colors[idx % len(default_pi_colors)]
            )

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=interval_df.index,
                    y=interval_df["upper"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    name=f"{interval_name}",
                )
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=interval_df.index,
                    y=interval_df["lower"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=interval_color,
                    name=f"{interval_name}",
                    showlegend=True,
                )
            )

    # Add forecast line
    if forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.iloc[:, 0],
                mode="lines",
                name="Point Forecast",
                line=dict(color=color_dict["forecast"], dash="dash"),
            )
        )

    # Add test data based on the selected mode
    if test_data is not None:
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data.iloc[:, 0],
                mode=test_data_mode,
                name="Test Data",
                marker=(
                    dict(color=color_dict["test"], size=6)
                    if test_data_mode == "markers"
                    else None
                ),
                line=(
                    dict(color=color_dict["test"])
                    if test_data_mode == "lines"
                    else None
                ),
            )
        )

    title = ""
    if include_title:
        title="Univariate Time Series Visualization"
    
    # Enable vertical spike lines on hover
    fig.update_layout(
        title=title,
        xaxis=dict(
            showspikes=True,  # Enable spikes
            spikemode="across",  # Spike spans across all traces
            spikesnap="cursor",  # Spike snaps to cursor position
            spikedash="dot",  # Style of spike line
            spikethickness=1.5,  # Thickness of spike line
            spikecolor="gray",  # Color of spike line
        ),
        yaxis=dict(showspikes=False),  # Disable horizontal spikes
        hovermode="x",  # Hover closest to x-axis value
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=y_axis_label,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    # Show the figure
    fig.show()

# def plot_time_series(
#     training_data: pd.DataFrame,
#     test_data: pd.DataFrame | None = None,
#     forecast: pd.DataFrame | None = None,
#     test_data_mode: str = "markers",
#     y_axis_label: str = "Value",
#     color_dict: dict | None = None,
# ) -> None:
#     """
#     Plots univariate time series data using Plotly. Options for displaying
#     training, test, and

#     Parameters:
#     ----------
#     training_data: pd.DataFrame
#         A pandas DataFrame with a DatetimeIndex containing the training data
#         It should have one column representing the time series values.

#     test_data: pd.DataFrame, optional (default=None)
#         A pandas DataFrame with a DatetimeIndex containing the test data.
#         Displayed as black dots or a line based on `test_data_mode`.

#     forecast: pd.DataFrame, optional (default=None)
#         A pandas DataFrame with a DatetimeIndex containing the forecasted data.

#     test_data_mode: str, optional (default='markers')
#         Mode for displaying test data.
#         Options are 'markers' (dots) or 'lines' (line).

#     y_axis_label: str, optional (default='Value')
#         The quantity measured by the time series to display on the y-axis

#     color_dict: dict, optional (default=None)
#         Dictionary with color codes for 'training', 'test', and 'forecast'.

#     Returns:
#         None: Displays an interactive Plotly figure.
#     """
#     # Validate test_data_mode input
#     if test_data_mode not in ["markers", "lines"]:
#         raise ValueError(
#             "Invalid value for test_data_mode. Choose 'markers' or 'lines'."
#         )

#     if color_dict is None:
#         color_dict = {
#             "training": "#0072B2",  # Default blue color for training data
#             "test": "#000000",  # Default black color for test data
#             "forecast": "#FF0000",  # Default red color for forecast data
#         }

#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add training data as a line plot
#     fig.add_trace(
#         go.Scatter(
#             x=training_data.index,
#             y=training_data.iloc[:, 0],
#             mode="lines",
#             name="Training Data",
#             line=dict(color=color_dict["training"]),
#         )
#     )

#     # Add test data based on the selected mode
#     if test_data is not None:
#         fig.add_trace(
#             go.Scatter(
#                 x=test_data.index,
#                 y=test_data.iloc[:, 0],
#                 mode=test_data_mode,
#                 name="Test Data",
#                 marker=(
#                     dict(color=color_dict["test"], size=6)
#                     if test_data_mode == "markers"
#                     else None
#                 ),
#                 line=(
#                     dict(color=color_dict["test"])
#                     if test_data_mode == "lines"
#                     else None
#                 ),
#             )
#         )

#     # Add forecast data as a line plot if provided
#     if forecast is not None:
#         fig.add_trace(
#             go.Scatter(
#                 x=forecast.index,
#                 y=forecast.iloc[:, 0],
#                 mode="lines",
#                 name="Point Forecast",
#                 line=dict(
#                     color=color_dict["forecast"], dash="dash"
#                 ),  # Dashed red line for forecast
#             )
#         )

#     # Enable vertical spike lines on hover
#     fig.update_layout(
#         title="Univariate Time Series Visualization",
#         xaxis=dict(
#             showspikes=True,  # Enable spikes
#             spikemode="across",  # Spike spans across all traces
#             spikesnap="cursor",  # Spike snaps to cursor position
#             spikedash="dot",  # Style of spike line
#             spikethickness=1.5,  # Thickness of spike line
#             spikecolor="gray",  # Color of spike line
#         ),
#         yaxis=dict(showspikes=False),  # Disable horizontal spikes
#         hovermode="x",  # Hover closest to x-axis value
#         template="plotly_white",
#         xaxis_title="Date",
#         yaxis_title=y_axis_label,
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     )

#     # Show the figure
#     fig.show()



