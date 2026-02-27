env_name="CARLPendulum"
export JAX_PLATFORMS=cpu

context_labels=("g" "l")
env_pf="pendulum"
alg_pf="ddpg"
prefix_experts="${env_pf}_${alg_pf}_sb3"
rel_std=(0.25 0.25)
context_min=(5.0 0.5)
context_max=(15.0 1.5) 
up_prefix="${env_pf}_${alg_pf}"
up_suffix="contexts_300"
stack_height=4
data_retain_frac=1.0
n_train_steps=500
lstm_suffix="dataretain_${data_retain_frac}"
OSI_hist=5
osi_layers=(128 32 32)
temp_suffix=$(IFS="_"; echo "${osi_layers[*]}")
osi_suffix="layers_${temp_suffix}_dataretain_${data_retain_frac}"
device="cuda"
out_path="outputs/pendulumout"

# python -u -m src.train_up_carl -carl_env_name ${env_name} -context_labels ${context_labels[@]} -prefix ${env_pf} -alg ${alg_pf} -n_train_samples 300 -exp_name "${up_suffix}" -rel_std ${rel_std[@]} -device ${device} -trainsteps 1000 > "${out_path}.output";
# python -u src/gen_evalset_up.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -context_min ${context_min[@]} -context_max ${context_max[@]} -distribution 'uniform' -up_prefix "${up_prefix}" -up_suffix ${up_suffix} -nrounds 3 -n_eval_samples 5 -eval_episodes 5 -device ${device} -n_parallel_processes 1 > "${out_path}2.output" ; # -nrounds 5 -n_eval_samples 100 -eval_episodes 10
# python -u src/train_pred_dagger.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -model_arch 'lstm' -up_prefix ${up_prefix} -up_suffix ${up_suffix} -dagger_rounds 2 -n_train_samples 300 -device ${device} -stack_height ${stack_height} -data_retain_frac ${data_retain_frac} -exp_name ${lstm_suffix} -n_train_steps ${n_train_steps} -max_epochs 2 > "${out_path}3.output"; # max_epochs remove -dagger_rounds 6 -n_train_steps remove
# python -u src/evaluate_up.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -context_min ${context_min[@]} -context_max ${context_max[@]} -prefix_experts ${prefix_experts} -all_suffix 'up_lstm_dag_1' -nrounds 3 -n_eval_samples 5 -nevals 1 -up_prefix ${up_prefix} -up_suffix ${up_suffix} -lstmsuffix ${lstm_suffix} -stack_height ${stack_height} -OSI_hist 10 -device ${device} -optimal_policy_name 'upc' > "${out_path}4.output";
# python -u src/train_pred_dagger.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -model_arch 'gru' -up_prefix ${up_prefix} -up_suffix 'contexts_300' -dagger_rounds 2 -n_train_samples 300 -device ${device} -stack_height ${stack_height} -data_retain_frac ${data_retain_frac} -exp_name ${lstm_suffix} -n_train_steps ${n_train_steps} -max_epochs 2 > "${out_path}5.output"; # max_epochs remove -dagger_rounds 6 -n_train_steps remove
# python -u src/evaluate_up.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -context_min ${context_min[@]} -context_max ${context_max[@]} -prefix_experts ${prefix_experts} -all_suffix 'up_gru_dag_1' -nrounds 3 -n_eval_samples 5 -nevals 1 -up_prefix ${up_prefix} -up_suffix ${up_suffix} -grusuffix ${lstm_suffix} -stack_height ${stack_height} -OSI_hist 10 -device ${device} -optimal_policy_name 'upc' > "${out_path}6.output";
# python -u -m src.train_osi_carl_ver3 -carl_env_name ${env_name} -context_labels ${context_labels[@]} -policy_prefix "${up_prefix}" -OSI_hist ${OSI_hist} -n_train_contexts 300 -up_suffix ${up_suffix} -exp_name ${osi_suffix} -osi_layers ${osi_layers[@]} -data_retain_frac ${data_retain_frac} -training_sample_num ${n_train_steps} -osi_iteration 2 > "${out_path}7.output"; #-osi_iteration remove -training_sample_num remove -n_train_contexts 300 -exp_name 'contexts_300'
python -u src/evaluate_up.py -carl_env_name ${env_name} -context_labels ${context_labels[@]} -context_min ${context_min[@]} -context_max ${context_max[@]} -prefix_experts ${prefix_experts} -all_suffix 'uposi' -nrounds 3 -n_eval_samples 5 -nevals 1 -up_prefix ${up_prefix} -up_suffix ${up_suffix} -OSI_hist ${OSI_hist} -device ${device} -osisuffix "_${osi_suffix}" -osi_layers ${osi_layers[@]} -optimal_policy_name 'upc' > "${out_path}6.output";
