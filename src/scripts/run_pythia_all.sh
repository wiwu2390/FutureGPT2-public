for SIZE in 14m 31m 70m 160m 410m 1b 1.4b 2.8b
do
    echo RUNNING VANILLA $SIZE
    python -m scripts.run_pythia $SIZE vanilla
    echo RUNNING MYOPIC $SIZE
    python -m scripts.run_pythia $SIZE myopic
done
