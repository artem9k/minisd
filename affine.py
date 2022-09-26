"""
Probability of Rain predictor
Uses 3 regressors:
- Percent cloudy
- Temperature (F)
- Days of rain in the past week

weight vector:
- Percent cloudy is given a weight of 25%, and divided by 100, since 
the fact that it is cloudy does not guarantee rain

- Temperature (F) is given a weight of 0.10, and divided by 100,
since warm weather means that it is more likely to rain then snow

- Days of rain in the past week is given a weight of 0.55, and divided
by 7, the number of days in a week, since if it has rained already, it is 
relatively likely to rain more

the bias will be 0.10, since there is always at least a small chance of rain.
"""

def weather_pred(x, b, a):
    dot = 0 
    for i in range(len(x)):
        dot += (x[i] * a[i])
    return dot + b

if __name__ == "__main__":
    weights = [0.0025, 0.001, 0.07857]
    bias = 0.1

    # a sunny, cold day with no rain in past week
    # output: 0.16 probability of rain
    print(weather_pred(weights, bias, [20, 15, 0]))

    # a cloudy, hot day with lots of previous rain
    # output: 0.76 probability of rain
    print(weather_pred(weights, bias, [75, 80, 5]))

