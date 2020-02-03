import math
from math import exp
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# class Bond for storing info about bond.
class Bond:
    coupon_rate = 0
    coupon_num = 0
    price = 0
    offset = 0
    face = 0

    def __init__(self, cr, cn, pr, f):
        self.coupon_num = cn
        self.coupon_rate = cr
        self.price = pr
        self.offset = 0
        self.face = f

    def increment_date(self, new_pr):
        self.offset += 1
        self.price = new_pr

    # calculate future value on the maturity date
    def get_fv(self, ytm):
        days_till_ma = (self.coupon_num-1)*184 + 59- self.offset
        c_days_till_m = (self.coupon_num-1)*184
        fv=0
        fv += self.price* exp(ytm*days_till_ma/365)
        for i in range(self.coupon_num-1):
            fv -= self.coupon_rate*self.face* exp(ytm*c_days_till_m/365)
            c_days_till_m -= 184
        fv -= self.coupon_rate*self.face+self.face
        return fv

    # calculate YTM by Newton Method
    def get_ytm(self):
        base = 1
        ytm = 0
        for decimal in range(8):

            for i in range(10):
                ytm += base
                if self.get_fv(ytm) >0:
                    ytm -= base
                    break
            base = base/10
        return math.exp(ytm)-1


# YTM
def ytm(df):
    c_num = 1
    ytm_list = []
    for i in range(100):
        if i % 10 == 0:
            temp = Bond(df['Coupon'][i]/200, c_num, df['Dirty'][i], 100)
            c_num += 1
        else:
            temp.increment_date(df['Dirty'][i])
        ytm_list.append(round(temp.get_ytm(), 4) * 100)

    df['YTM'] = [round(x,2) for x in ytm_list]


# Calculate dirty price of bonds
def dirty_price(df):
    ai = []
    dp = []
    for i in range(100):
        day_diff = 59 - (i % 10)
        ai.append(round(((184 - day_diff) / 365) * df['Coupon'][i], 2))
        dp.append(df['Price'][i] + ai[i])

    df['Accrued'] = [round(x,2) for x in ai]
    df['Dirty'] = [round(x,2) for x in dp]


# Plot the YTM curves
def plot_ytm(df):
    ytm_y = []
    temp = []
    for i in range(10):
        for j in range(100):
            if df['Day'][j] == i+1:
                temp.append(round(df['YTM'][j],2))
        ytm_y.append(temp)
        temp = []

    term = [1,2,3,4,5,6,7,8,9,10]

    for i in range(len(ytm_y)):
        plt.plot(term, ytm_y[i])

    plt.title('Yield to Maturity of 10 bonds')
    plt.xlabel('number of terms(semiannual)')
    plt.ylabel('YTM(%)')
    plt.legend(labels = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10'], loc = 'best')
    plt.show()


# Calculate spot rates
def spot_rate(df):
    bd1_spot = []
    day_diff = 59
    for i in range(10):
        spr = - math.log(df['Dirty'][i]/ (df['Coupon'][i]/2 + 100) ) / (day_diff / 365)
        bd1_spot.append(round(spr,4))
        day_diff -= 1

    result = [bd1_spot]

    for i in range(1, 10):
        temp = []
        for j in range(10):
            index = i * 10 + j
            pmt = 0
            coupon_num = i
            while coupon_num > 0:
                prev_sp = result[coupon_num - 1][j]
                days = (coupon_num - 1) * 184 + 59 - j
                pmt += df['Coupon'][index]/2 * math.exp(-prev_sp * days / 365)
                coupon_num -= 1
            spot = - math.log((df['Dirty'][index] - pmt)/(df['Coupon'][index]/2+100)) / ((i*184+59-j)/365)
            temp.append(round(spot,4))
        result.append(temp)

    write_sp = []
    for i in range(len(result)):
        write_sp += result[i]
    df['Spot'] = write_sp
    return result


# plot the spot curves
def plot_spot(spot_rates):
    term = [1,2,3,4,5,6,7,8,9,10]

    m = np.array(spot_rates)
    m_tr = m.transpose()
    for i in range(10):
        plt.plot(term, m_tr[i])

    plt.title('Spot Rates of 10 bonds')
    plt.xlabel('number of terms(semiannual)')
    plt.ylabel('Spot Rate')
    plt.legend(labels=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'], loc='best')
    plt.show()


# calculate forward rates and plot the forward rate curves
def forward_rate(df):
    spr_base_on_bond = []
    temp = []
    for i in range(100):
        if i % 10 == 0:
            if len(temp) != 0:
                spr_base_on_bond.append(temp)
                temp = []
        temp.append(df['Spot'][i])
    spr_base_on_bond.append(temp)

    spr_day = np.array(spr_base_on_bond).transpose()

    for i in range(0,5):
        spr_day = np.delete(spr_day, i+1, 1)

    forward = [[] for i in range(10)]
    for i in range(10):
        for j in range(1,5):
            r_j = spr_day[i][j]
            r_0 = spr_day[i][0]
            t_j = (59 - i + 184 * j * 2) / 365
            t_0 = (59 - i) /365
            forward[i].append(round((r_j * t_j - r_0 * t_0) / (t_j - t_0),4))

    fwd_term = [2,3,4,5]
    for i in range(len(forward)):
        plt.plot(fwd_term, forward[i])

    plt.title('Forward Rates of 10 bonds')
    plt.xlabel('Terms range from 1yr')
    plt.ylabel('Forward Rate')
    plt.legend(labels=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'], loc='best')
    plt.show()
    return forward


# Get covariance matrices of daily log returns and forward rates.
def cov_matrix(df, forward):
    A = df['YTM'][10:20].to_list()
    A = ['%.2f' % elem for elem in A]
    B = df['YTM'][30:40].to_list()
    B = ['%.2f' % elem for elem in B]
    C = df['YTM'][50:60].to_list()
    C = ['%.2f' % elem for elem in C]
    D = df['YTM'][70:80].to_list()
    D = ['%.2f' % elem for elem in D]
    E = df['YTM'][90:100].to_list()
    E = ['%.2f' % elem for elem in E]

    X = pd.DataFrame({'A':A,'B':B,'C':C,'D':D,'E':E}).to_numpy()

    result = [[],[],[],[],[]]

    for i in range(5):
        for j in range(9):
            result[i].append(round(math.log( float(X[j+1][i]) / float(X[j][i]) ),4))

    cov_of_log_returns = np.cov(result)
    print(cov_of_log_returns)

    forward = np.array(forward).transpose()
    cov_of_forward = np.cov(forward)
    print(cov_of_forward)

    return (cov_of_log_returns, cov_of_forward)


# Calculate covariance matrices' eigenvalues and eigenvectors.
def find_e_value_vector(t):
    cov_log = t[0]
    cov_fwd = t[1]
    log_e_value, log_e_vector = np.linalg.eig(cov_log)
    fwd_e_value, fwd_e_vector = np.linalg.eig(cov_fwd)
    print(log_e_value)
    print(log_e_vector)
    print(fwd_e_value)
    print(fwd_e_vector)


if __name__ == "__main__":
    df = pd.read_csv('10DayPrices.csv')
    dirty_price(df)
    ytm(df)
    spt_rates = spot_rate(df)
    forward = forward_rate(df)

    # Write all info back to original file
    df.to_csv('info_included.csv')

    tuple_cov = cov_matrix(df, forward)
    find_e_value_vector(tuple_cov)

    plot_ytm(df)
    plot_spot(spt_rates)





