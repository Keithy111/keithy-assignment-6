from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # STEP 1: Generate random dataset X of size N with values between 0 and 1
    X = np.random.rand(N, 1)  # Generate N random values between 0 and 1 and reshape for sklearn
    # Generate Y values with normal additive error (mean mu, variance sigma^2)
    Y = 3 * X + mu + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))  # Adjust for target slope 3, with noise

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X, Y)
    slope = model.coef_[0][0]  # Get slope (coefficient) of the model
    intercept = model.intercept_[0]  # Get intercept of the model

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data points")
    plt.plot(X, model.predict(X), color="red", label=f"Regression line: Y = {slope:.2f}X + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter plot with Regression Line")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations and create histograms of slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate random X and Y values with same parameters
        X_sim = np.random.rand(N, 1)
        Y_sim = 3 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))
        
        # Fit a linear regression model to X_sim and Y_sim
        sim_model = LinearRegression().fit(X_sim, Y_sim)
        slopes.append(sim_model.coef_[0][0])  # Append slope
        intercepts.append(sim_model.intercept_[0])  # Append intercept

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")



