export HYDRA_FULL_ERROR=1

python /home/foresti/infosedd/evaluation/run.py --config-name config\
 +estimator=infosedd\
 +distribution=categorical_long_vector\
 ++distribution.mutual_information=0.5\
 ++estimator.args.variant=j\
 ++estimator.args.use_marginal_flag=False\
 ++estimator.args.is_parametric_marginal=False\
 ++estimator.args.training.batch_size=1024\
 ++estimator.args.training.max_steps=1e4\
 ++estimator.args.optim.lr=1e-2\
 ++estimator.args.training.devices=[2]\
 ++distribution.seq_length=2