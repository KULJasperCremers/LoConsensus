import old_files.timeseries_generator as tsg
import old_files.visualize as vis
import parallelization as par
import path as path
import similarity_matrix as sm

ts1 = tsg.TimeSeries(100, [1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
ts2 = tsg.TimeSeries(100, [1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
sm1 = sm.SimilarityMatrix(ts1.ts, ts2.ts)
d1 = sm1.generate_csm(sm1.sm)
sm2 = sm.SimilarityMatrix(ts2.ts, ts1.ts)
d2 = sm2.generate_csm(sm2.sm)

# par.plot_parallel(vis.plot_sm, (ts1.ts, ts2.ts, sm1.sm), {'matshow_kwargs': {'alpha': 0.33}},
# vis.plot_sm, (ts2.ts, ts1.ts, sm2.sm), {'matshow_kwargs': {'alpha': 0.33}})

p = path.Path(l_min=1, l_max=10)
paths = p.find_best_paths(d1)
print(paths)
