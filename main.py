from functions import *


# Получаем общую информацию о командах
df = {}
for state in data_states:
    df[state] = get_data('source/shots', general_info_columns)

# Получаем нужные параметры
for op in options:
    df['inital'][op] = get_data(f'source/{op}', columns_in_options[op])
    df['stand'][op] = standardize(get_data(f'source/{op}', columns_in_options[op]), op)
    df['norm'][op] = normalize(get_data(f'source/{op}', columns_in_options[op]), op)

# Сохраняем описательные статистики изначальных данных в виде таблицы, чтобы было проще копировать в отчет
# write_describe_to_excel(df['inital'])

# Сохраняем диаграммы рассеяния попарно между всеми показателями
# for ops in itertools.combinations(options, 2):
#     plot_scatter(ops, df['inital'])

# Сохраняем дендрограммы
# for state in data_states:
#     plot_dend(df[state], dend_names[state])

# Метод локтя
# plot_elbow_method(df)

# Сохраняем график средних силуэтов
# plot_avg_silhouette(df)

# Сохраняем графики силуэтов кластеров
# plot_silhouette(df)

# Кластеризуем
# for state in data_states[1:]:
#     cluster(df[state], 5, state)
