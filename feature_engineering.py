from data_transformation import df_temp
import numpy as np
import pandas as pd
import seaborn
import os
import sys
from scipy.stats import pearsonr
import statsmodels.api

# Ignore this function for now, still working on it
def continuous_continuous_pairs(df, response, cont_predictors):
    # Correlation Matrix for cont-cont

    s = seaborn.heatmap(
        df[cont_predictors].corr(),
        annot=True,
        vmax=1,
        vmin=-1,
        center=0,
        cmap="vlag",
    )
    s = s.get_figure()
    s.savefig("Continuous_Continuous_correlation_matrix.png")

    # Correlation table logic
    corr_values = pd.DataFrame(
        columns=["predictor_1", "predictor_2", "corr_value", "abs_corr_value"]
    )
    temp_corr = []
    table = pd.DataFrame(columns=["predictor", "file_link"])

    for i in cont_predictors:

        for j in cont_predictors:
            if i == j:
                continue
            else:
                corr, _ = pearsonr(df[i], df[j])

                if corr not in temp_corr:
                    temp_corr.append(corr)
                    corr_values.loc[len(corr_values)] = [i, j, corr, abs(corr)]

        predictor_name = statsmodels.api.add_constant(df[i])
        y = df[response]
        y, temp_var = y.factorize()

        linear_regression_model = statsmodels.api.Logit(y, predictor_name)
        linear_regression_model_fitted = linear_regression_model.fit()
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        # print(f"t value: {t_value}", f"p value: {p_value}")

        fig = px.scatter(x=df[i], y=df[response], trendline="ols")
        fig.update_layout(
            title=f"Variable: {i}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {i}",
            yaxis_title=f"{response}",
        )
        filename = f"{response}_{i}_plot.html"
        fig.write_html(file=filename, include_plotlyjs="cdn")

        table.loc[len(table)] = [i, filename]

    table["file_link"] = os.getcwd() + "/" + table["file_link"]
    temp_table = pd.merge(
        corr_values, table, how="inner", left_on="predictor_1", right_on="predictor"
    )
    final_table = pd.merge(
        temp_table, table, how="inner", left_on="predictor_2", right_on="predictor"
    )
    final_table = final_table[
        [
            "predictor_1",
            "predictor_2",
            "corr_value",
            "abs_corr_value",
            "file_link_x",
            "file_link_y",
        ]
    ]
    final_table = final_table.rename(
        columns={
            "file_link_x": "regression_plot_link_predictor_1",
            "file_link_y": "regression_plot_link_predictor_2",
        }
    )
    final_table.sort_values(by="corr_value", ascending=False, inplace=True)

    final_table["regression_plot_link_predictor_1"] = final_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["regression_plot_link_predictor_1"], x["regression_plot_link_predictor_1"]
        ),
        axis=1,
    )
    final_table["regression_plot_link_predictor_2"] = final_table.apply(
        lambda x: '<a href="{}">{}</a>'.format(
            x["regression_plot_link_predictor_2"], x["regression_plot_link_predictor_2"]
        ),
        axis=1,
    )
    final_table.to_html(
        "Continuous continuous rankings and regression plot.html",
        escape=False,
        render_links=True,
    )

def check_response_type(response):
    if len(set(response)) < 3:
        var = "Categorical"
        print("Response variable is boolean!")
    else:
        var = "Continuous"
        print("Response variable is continuous!")
    return var


def check_predictor_type(predictor):
    if predictor.dtype == "object":
        return "Categorical"
    else:
        return "Continuous"

def main():


    # In order to increase the number of predictors, I'm splitting the genre column into boolean values (0 or 1) for every genre
    df_temp['Action'] = df_temp['genres'].str.contains('Action')
    df_temp['Adventure'] = df_temp['genres'].str.contains('Adventure')
    df_temp['Animation'] = df_temp['genres'].str.contains('Animation')
    df_temp['Children'] = df_temp['genres'].str.contains('Children')
    df_temp['Comedy'] = df_temp['genres'].str.contains('Comedy')
    df_temp['Crime'] = df_temp['genres'].str.contains('Crime')
    df_temp['Documentary'] = df_temp['genres'].str.contains('Documentary')
    df_temp['Drama'] = df_temp['genres'].str.contains('Drama')
    df_temp['Fantasy'] = df_temp['genres'].str.contains('Fantasy')
    df_temp['Film-Noir'] = df_temp['genres'].str.contains('Film-Noir')
    df_temp['Horror'] = df_temp['genres'].str.contains('Horror')
    df_temp['Musical'] = df_temp['genres'].str.contains('Musical')
    df_temp['Mystery'] = df_temp['genres'].str.contains('Mystery')
    df_temp['Romance'] = df_temp['genres'].str.contains('Romance')
    df_temp['Sci-Fi'] = df_temp['genres'].str.contains('Sci-Fi')
    df_temp['Thriller'] = df_temp['genres'].str.contains('Thriller')
    df_temp['War'] = df_temp['genres'].str.contains('War')
    df_temp['Western'] = df_temp['genres'].str.contains('Western')


    # Creating a predictor having logarithmic values for our response variable (lifetime_gross)
    df_temp['log_revenue'] = np.log1p(df_temp['lifetime_gross'])

    '''As of now, we have 'lifetime_gross' as response, 
    and predictors are ['avg_rating','log_revenue', 'year' ,'studio', 'Action', 'Adventure', 'Animation',
           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
           'Thriller', 'War', 'Western']
    '''

    response = 'lifetime_gross'
    predictors = ['avg_rating','log_revenue','year' ,'studio', 'Action', 'Adventure', 'Animation',
           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
           'Thriller', 'War', 'Western']

    print(check_response_type(df_temp[response]))

    # List of categorical and continuous predictors separated in two lists
    cat_predictors = []
    cont_predictors = []

    for i in predictors:
        if check_predictor_type(df_temp[i]) == "Categorical":
            cat_predictors.append(i)
        elif check_predictor_type(df_temp[i]) == "Continuous":
            cont_predictors.append(i)

    print("Categorical predictors are -", cat_predictors)
    print("Continuous predictors are -", cont_predictors)

    '''Since we have two separate lists for categorical and continuous predictors, we can now perform some analysis
    on combination of categorical-categorical, categorical-continuous, continuous-continuous predictor pairs.
    The function at the beginning of this file would help us do analysis between continuous-continuous pairs, likewise,
    I'm planning to build functions for categorical-categorical, categorical-continuous pairs
    '''


    '''What if instead of predicting lifetime_gross, we cut the lifetime_gross variable into 50 bins, and then try
    to predict the bin where the movie should belong? More like 50 classifications, than regression. 
    '''
    response_bins = pd.cut(df_temp['lifetime_gross'], bins=50)




if __name__ == "__main__":
    sys.exit(main())
