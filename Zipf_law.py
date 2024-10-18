import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def read_file(filename: str):
    with open(filename, encoding='utf-8') as f:
        content = f.read()
        words_list = re.split(r"[\b\W\b]+", content)
        dictionary = {}
        for word in words_list:
            if word in dictionary.keys():
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        dictionary_sorted= dict(sorted (dictionary.items(), key= lambda x:x[1], reverse=True))
        return dictionary_sorted


def create_file_with_stats(filename: str, dictionary_sorted: dict):
        stats = []
        r = 0
        previous_frequency = 0
        total_number_of_words = sum(dictionary_sorted.values())
        for word in dictionary_sorted:
            frequency = dictionary_sorted[word]/total_number_of_words
            if frequency != previous_frequency:
                r += 1
            previous_frequency = frequency
            stats.append({'rank': r, 'word': word, 'count': dictionary_sorted[word], 'frequency': frequency})

        book_title = filename.split('.txt')[0]
        stats = pd.DataFrame(stats)
        plot(stats, book_title)
        new_file_name = book_title + str(total_number_of_words) + '.txt'
        stats.to_csv(new_file_name, sep='\t')
        return np.array(stats.index)


def read_and_create_file_with_stats(filename: str):
    return create_file_with_stats(filename, read_file(filename))


def plot(stats: pd.DataFrame, book_title: str):

    x_th, y_th = create_theoretical_zipf_distribution(stats)

    x = stats['rank']
    y = stats['frequency']
    plt.scatter(x, y, color='blue', s=2, label=book_title)
    plt.scatter(x_th, y_th, color='red', s=0.5, label="Zipf's law")
    plt.title(book_title)
    plt.xlabel('rank')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()
    plt.show()

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y, color='blue', label=book_title)
    plt.plot(x_th, y_th, color='red', label="Zipf's law")
    plt.title(book_title)
    plt.xlabel('rank')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()
    plt.show()


def plot_optimal_parameters(optimal_params: dict):
    for optimal_param in optimal_params:
        x, y = optimal_params[optimal_param]
        plt.scatter(x, y, s=8, label=optimal_param.split('.txt')[0])
    plt.xlabel('b parameter')
    plt.ylabel('a parameter')
    plt.legend()
    plt.grid()
    plt.show()


def create_theoretical_zipf_distribution(stats: pd.DataFrame):
    th_values = pd.DataFrame()
    ref_one = 1/sum(1/stats['rank'])
    print(ref_one)
    th_values['rank'] = stats['rank']
    # for 0.1 curves overlay much more accurate than for ref_one
    th_values['frequency'] = ref_one/th_values['rank']
    return th_values['rank'], th_values['frequency']


def zipf_mandelbrot_law(r, b, a):
    return 1/(r + b)**a


def optimal_parameters(stats_rank: np.array):
    rdata = stats_rank
    f = zipf_mandelbrot_law(rdata, 1, 2)
    rng = np.random.default_rng()
    f_noise = 0.2 * rng.normal(size=rdata.size)
    fdata = f + f_noise
    popt, pcov = curve_fit(zipf_mandelbrot_law, rdata, fdata)
    print(popt)
    return popt


def main():
    books = ['emma.txt', 'persuasion.txt', 'pride_and_prejudice.txt','sense_and_sensibility.txt']
    optimal_params = {}
    for book in books:
        stats_rank = read_and_create_file_with_stats(book)
        optimal_params[book] = optimal_parameters(stats_rank)
    plot_optimal_parameters(optimal_params)

if __name__ == '__main__':
    main()