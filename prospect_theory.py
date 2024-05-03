"""Implement basic prospect theory curves based on partial sigmoid curves."""

import numpy as np
import matplotlib.pyplot as plt

def curve_fit(expected_values: list[float], attitudes: list[float]) -> tuple[np.ndarray, np.ndarray]:
  """Fit data with expected values and attitudes towards risky gambles, and return a dictionary of"""
  gains = np.array([expected_value for expected_value, attitude in zip(expected_values, attitudes) if np.greater_equal(expected_value, 0)])
  g_attitude = np.array([attitude for expected_value, attitude in zip(expected_values, attitudes) if np.greater_equal(expected_value, 0)])
  losses = np.array([expected_value for expected_value, attitude in zip(expected_values, attitudes) if np.less(expected_value, 0)])
  l_attitude = np.array([attitude for expected_value, attitude in zip(expected_values, attitudes) if np.less(expected_value, 0)])

  from scipy.optimize import curve_fit

  def sigmoid(x, L ,x0, k, b):
      y = L / (1 + np.exp(-k*(x-x0))) + b
      return (y)

  # Fit for gains
  p0 = [max(g_attitude), np.median(gains),1,min(g_attitude)] # this is an mandatory initial guess
  popt, _ = curve_fit(sigmoid, gains, g_attitude,p0, method='dogbox', maxfev=100000)

  # Fit for losses
  q0 = [max(l_attitude), np.median(losses),1,min(l_attitude)] # this is an mandatory initial guess
  qopt, _ = curve_fit(sigmoid, losses, l_attitude,q0, method='dogbox', maxfev=100000)

  l_x = np.linspace(-10,0,100)
  g_x = np.linspace(0,10,100)


  x = np.concatenate(
     (l_x, g_x)
  )
  curve = np.concatenate(
     (sigmoid(l_x, *qopt),
     sigmoid(g_x, *popt))
  )


  return x, curve

def plot_curve(
    x: np.ndarray,
    curve: np.ndarray,
    expected_values: list[float],
    attitudes: list[float],
    title: str = "Risky Gamble Value Estimates"
) -> None:
  """Plot a prospect theory curve."""
  plt.plot(x, curve, '--k')
  plt.plot(expected_values, attitudes, 'yo')
  plt.xlabel("Expected Value")
  plt.ylabel("Affective Value")
  plt.ylim(-1., 1)
  plt.title(title)
  plt.show()

   
