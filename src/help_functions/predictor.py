import pandas as pd
from gpytorch import Module
from gpytorch.likelihoods import Likelihood
from scipy.stats import norm
import torch
import plotly

def predict_with_trained_model(model: Module, likelihood: Likelihood, X_test: torch.Tensor):
    """
    Predicts the output using the trained model and likelihood.

    Args:
        model: The trained model.
        likelihood: The likelihood function.
        X_test: The input data for prediction.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get the predictive distribution
        predictive_dist = likelihood(model(X_test))

        # Get the mean and variance of the predictions
        mean = predictive_dist.mean
        variance = predictive_dist.variance

    return mean, variance

def create_df_from_predictions(mean, variance, indexing):
    """
    Creates a DataFrame from the predictions.

    Args:
        mean: The mean predictions.
        variance: The variance of the predictions.
        indexing: The index for the DataFrame.
    """
    predictions_df = pd.DataFrame({
        'mean': mean.numpy(),
        'variance': variance.numpy()
    }, index=indexing)

    return predictions_df

def plot_predictions(predictions_df, title="Predictions", confidence=0.95):
    """
    Plots the predictions using Plotly.

    Args:
        predictions_df: DataFrame containing the mean and variance of predictions.
        title: Title of the plot.
        confidence: Confidence level for the shaded interval (default 0.95 for a 95% CI).
    """
    fig = plotly.graph_objects.Figure()

    # Add mean predictions
    fig.add_trace(plotly.graph_objects.Scatter(
        x=predictions_df.index,
        y=predictions_df['mean'],
        mode='lines',
        name='Mean Prediction'
    ))

    # Compute the confidence interval from the standard deviation
    z = norm.ppf(0.5 + confidence / 2)
    std = predictions_df['variance'] ** 0.5
    upper = predictions_df['mean'] + z * std
    lower = predictions_df['mean'] - z * std

    x_index = predictions_df.index.to_series()

    # Add confidence interval as shaded area
    fig.add_trace(plotly.graph_objects.Scatter(
        x=pd.concat([x_index, x_index[::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{int(confidence * 100)}% CI'
    ))

    fig.update_layout(title=title,
                      xaxis_title='Index',
                      yaxis_title='Predicted Value',
                      template='plotly_white')

    fig.show()