from sklearn.ensemble import RandomForestRegressor
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib as mlp
# this is used because normally by default matplotlib uses tkinter graphics backend
# but on headless servers there are no tkinter libraries, so this tells matplotlib
# to use it's own backend
mlp.use('Agg')
from matplotlib import pyplot as plt
import locale
import time

# for DB connection
import pymysql
from sqlalchemy import create_engine
db_conx = create_engine('mysql+pymysql://root:abhinav@7@localhost:3306/prediction_output', echo=False)

# data contains some integer strings in US locale format, like "1,234", that needs to be converted to 1234
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

holiday_list_file = 'us_holidays.csv'
filename = "ML_NEW_IP.csv"


# this method takes in the filename and outputs the dataframe in time-series format
def build_data_frame(data_frame, h):

    length_df = data_frame.__len__()

    # seperate the data which will be used for predictions
    prediction_input_frame = data_frame.loc[length_df-h-1:]

    # drop the data which will be used in prediction
    data_frame = data_frame.drop(data_frame.index[length_df-h:])

    # convert the string in the format of US locale format that is 1,029 strings into proper integers
    sales_data = [x if type(x) != str else locale.atoi(x) for x in data_frame.sales]
    budget_data = [x if type(x) != str else locale.atoi(x) for x in data_frame.budget]
    data_frame.budget = pd.DataFrame(budget_data).values

    data_frame.sales = pd.DataFrame(sales_data).values

    # make all of the data as float64
    data_frame[['sales', 'promo', 'budget', 'retail', 'holiday']] = data_frame[
        ['sales', 'promo', 'budget', 'retail', 'holiday']
    ].apply(
        pd.to_numeric
    )

    # for dropping the entire row containing NaN values
    # print data_frame.dropna()

    # for selecting rows containing finite values
    # data_frame = data_frame[np.isfinite(data_frame[['sales', 'budget', 'promo', 'retail']])]

    # to fill the NaN values with the mean value of that column
    data_frame = data_frame.fillna(value=data_frame.mean())

    # this prints the mean of the each column of the dataframe
    # print data_frame.mean()

    if h < 1:
        return
    elif h >= data_frame.__len__() / 2:
        return
    else:
        for i in range(h):
            prediction_input_frame['sales' + str(i)] = prediction_input_frame.sales.shift(i+1)
            prediction_input_frame['promo' + str(i)] = prediction_input_frame.promo.shift(i+1)
            prediction_input_frame['retail' + str(i)] = prediction_input_frame.retail.shift(i+1)
            prediction_input_frame['budget' + str(i)] = prediction_input_frame.budget.shift(i+1)
            prediction_input_frame['holiday' + str(i)] = prediction_input_frame.holiday.shift(i + 1)
            data_frame['sales' + str(i)] = data_frame.sales.shift(i+1)
            data_frame['promo' + str(i)] = data_frame.promo.shift(i+1)
            data_frame['retail' + str(i)] = data_frame.retail.shift(i+1)
            data_frame['budget' + str(i)] = data_frame.budget.shift(i+1)
            data_frame['holiday' + str(i)] = data_frame.holiday.shift(i + 1)
        new_data_frame = data_frame.drop(data_frame.index[:h//2])

        new_data_frame = new_data_frame.drop(new_data_frame.index[:new_data_frame.first_valid_index()+1])

    prediction_input_frame = prediction_input_frame[-1:]

    return new_data_frame, data_frame, prediction_input_frame


def holiday_mapping(holiday_list_file):
    holiday_data_frame = pd.read_csv(holiday_list_file)
    holiday_data_frame.date = pd.to_datetime(holiday_data_frame.date, format="%Y-%m-%d")
    list_of_unique_holidays = holiday_data_frame.holiday.unique()

    holiday_dict = {}
    holiday_id = []

    for index, i in enumerate(list_of_unique_holidays):
        holiday_dict[i] = index + 1
    for i in holiday_data_frame.holiday:
        if i in holiday_dict.keys():
            holiday_id.append(holiday_dict[i])

    holiday_data_frame['holiday'] = np.array(holiday_id)
    return holiday_data_frame


def separate_multiple_products(data_input_filename, h):
    data_frame_dict = {}
    data_frame = pd.read_csv(data_input_filename)
    holiday_data_frame = holiday_mapping(holiday_list_file)

    # make all the labels of the columns in lower case
    data_frame.columns = map(str.lower, data_frame.columns)

    # make all of the entries in the variables column as lower case
    data_frame.variables = data_frame['variables'].str.lower()

    # drop these fields to be left only with the weeks
    data_frame2 = data_frame.drop(['cid', 'region', 'pid', 'lid', 'variables'], axis=1)
    count = 0
    last_input_week = data_frame2.keys()[-1-h]
    last_output_week = data_frame2.keys()[-1]
    first_input_week = data_frame2.keys()[0]

    try:
        last_input_date = datetime.datetime.strptime(last_input_week + '-1', '%yw%W-%w')
        last_output_date = datetime.datetime.strptime(last_output_week + '-1', '%yw%W-%w')
        first_input_date = datetime.datetime.strptime(first_input_week + '-1', '%yw%W-%w')
    except:
        last_input_date = datetime.datetime.strptime(last_input_week + '-1', 'w%W %Y-%w')
        last_output_date = datetime.datetime.strptime(last_output_week + '-1', 'w%W %Y-%w')
        first_input_date = datetime.datetime.strptime(first_input_week + '-1', 'w%W %Y-%w')

    dates = [(first_input_date + timedelta(7 * i)) for i in range((last_output_date-first_input_date).days//7+2)]

    # create a dict for each unique product entry
    for i, j, k, l, m in zip(data_frame.pid, data_frame.region, data_frame.lid, data_frame.cid, data_frame.variables):
        key = ','.join((map(str, [i, j, k, l])))
        # if key exits just add it as any other entry to dict
        if key in data_frame_dict.keys():
            data_frame_dict[key][m] = list(data_frame2.loc[count].values)
        # if key already not there, create a dict entry first
        else:
            data_frame_dict[key] = {m: list(data_frame2.loc[count].values)}
        count += 1
    for key in data_frame_dict.keys():
        sales_data = data_frame_dict[key]
        # now separate the each product into separate data frame
        try:
            new_data_frame = pd.DataFrame(
                {'promo': sales_data['promo'], 'retail': sales_data['retail price'],
                 'budget': sales_data['budget'], 'sales': sales_data['sales']})
        except:
            new_data_frame = pd.DataFrame(
                {'promo': sales_data['promo'], 'retail': sales_data['retail'],
                 'budget': sales_data['budget'], 'sales': sales_data['sales']})
        # print(key)
        new_data_frame['holiday'] = np.array([0 for i in range(new_data_frame.__len__())])

        new_data_frame['date'] = np.array(dates)

        for index, date in enumerate(holiday_data_frame.date):

            for i, j in enumerate(new_data_frame.date):

                var = j - date
                if var <= timedelta(4) and var >= timedelta(-1):
                    new_data_frame.loc[i, 'holiday'] = holiday_data_frame.loc[index, 'holiday']

        new_data_frame = new_data_frame.drop(['date'], axis=1)
        yield new_data_frame, key.split(','), data_frame.keys()[:4].values, last_input_date.date(), first_input_date.date()


def model(filename, h, criterion='mse', trees=10):
    output_frames = []
    data_frames = []
    count = 0
    feature_importances = []
    predictions = []
    accuracy_scores = []
    for frame, lables_values, labels, last_date, first_date in separate_multiple_products(filename, h):

        new_data_frame, data_frame, prediction_input_frame = build_data_frame(frame, h)

        data_frames.append(data_frame)
        clf = RandomForestRegressor(n_estimators=trees, criterion=criterion)

        x_drop_list = []
        y_drop_list = []
        x_drop_list.append('sales')
        for i in range(h-1):
            x_drop_list.append('sales' + str(i))
        for i in new_data_frame.keys():
            if i in x_drop_list:
                pass
            else:
                y_drop_list.append(i)
        x_predict = prediction_input_frame.drop(x_drop_list, axis=1)

        x_train = new_data_frame.drop(x_drop_list, axis=1)

        y_train = new_data_frame.drop(y_drop_list, axis=1)

        clf.fit(x_train, y_train)

        prediction = (clf.predict(x_predict))
        predictions.append(prediction)

        accuracy_scores.append(clf.score(x_train, y_train))
        feature_importances.append(clf.feature_importances_)

        output_frames.append(build_output_frame(prediction, labels, lables_values, last_date, h))
        count += 1
    avg_acc_score = sum(accuracy_scores)/len(accuracy_scores)

    sale = 0
    budget = 0
    retail = 0
    promo = 0
    holiday = 0

    feature_importances = map(sum, zip(*feature_importances))

    for i, m in zip(feature_importances, new_data_frame.keys().values):
        if 'sale' in m:
            sale += i
        elif 'budget' in m:
            budget += i
        elif 'retail' in m:
            retail += i
        elif 'promo' in m:
            promo += i
        elif 'holiday' in m:
            holiday += i

    graph_data = []
    for i in range(count):
        graph_data.append(data_frames[i].sales.tolist() + predictions[i].tolist()[0])

    fig, ax2 = plt.subplots(figsize=(10, 6))

    ax2.pie(
        [sale * 100 / count, budget * 100 / count, promo * 100 / count,
         retail * 100 / count, holiday * 100 / count],
        # explode=explode,
        labels=['Sale', 'Budget', 'Promo', 'Retail Price', 'Holiday'],
        autopct='%1.1f%%',
        shadow=True,
        startangle=90)
    ax2.axis('equal')

    plt.title('Accuracy Metrics({0:.2f})'.format(avg_acc_score*100))

    plt.savefig('accuracy.png')

    final_data_frame = pd.concat(output_frames, ignore_index=True)
    legend_labels = []

    for i in output_frames:
        legend_labels.append('pid:{0},region:{1},lid:{2},cid:{3}'.format(i.pid[0], i.region[0], i.lid[0], i.cid[0]))

    xtick_labels = [(first_date + timedelta(7 * i)).__str__() for i in range(1, h + len(data_frame.sales) + 1)]

    plot_graphs(
        graph_data,
        title='Sales Forecasting',
        x_label='Weeks',
        y_label='Sales Quantity',
        xtick_labels=xtick_labels,
        legend_labels=legend_labels,
        filename='sales.png'
    )
    time_stamp = time.strftime("%d/%m/%Y").__str__().replace('/', '') + \
                 time.strftime("%H:%M:%S").__str__().replace(':', '')
    # final_data_frame.to_csv('output.csv', index=False)
    feature_importances_df = pd.DataFrame([[sale * 100 / count, budget * 100 / count, promo * 100 / count,
                                            retail * 100 / count, holiday * 100 / count, avg_acc_score * 100]],
                                          columns=['sale', 'budget', 'promo', 'retail', 'holiday', 'Accuracy'])
    feature_importances_df.to_sql(name='FeatureImportance',
                                  con=db_conx,
                                  if_exists='append',
                                  index=True,
                                  index_label=time_stamp)
    final_data_frame.to_sql(name=time_stamp,
                            con=db_conx,
                            if_exists='replace',
                            index=False)
    return


def build_output_frame(predictions, labels, labels_values, last_date, h):
    output_frame = pd.DataFrame([
        [int(labels_values[0]), int(labels_values[1]), int(labels_values[2]), int(labels_values[3])]
    ],
        columns=[labels[0], labels[1], labels[2], labels[3]])

    # column label as dates for upcoming weeks
    columns = [(last_date + timedelta(7*i)).__str__() for i in range(1, h + 1)]

    if h == 1:
        prediction_frame = pd.DataFrame(predictions, columns=columns)
    else:
        prediction_frame = pd.DataFrame(np.fliplr(predictions), columns=columns)
    prediction_frame = prediction_frame.astype(int)
    output_frame = output_frame.join(other=prediction_frame, how='right')
    return output_frame


def plot_graphs(data, title='Graph', x_label='X-axis', y_label='Y-axis', xtick_labels=[], legend_labels=[], filename='fig.png'):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80, squeeze=True)
    x_axis_len = len(data[0])
    total_plots = len(data)
    if total_plots == 1:
        ax.plot(data[0])
    elif total_plots > 1:
        for i in range(total_plots):
            ax.plot(data[i], label=legend_labels[i])

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    step = len(data[0]) // 22
    if step >= 10:
        pass
    else:
        step = 1
    ax.set_xticks(np.arange(x_axis_len)[::step])
    ax.set_xticklabels(xtick_labels[::step])
    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()

    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    ax.legend()
    # plt.show()
    plt.savefig(filename)


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()//2., 1.005*height,
                '%d' % int(height),
                ha='center', va='bottom')
