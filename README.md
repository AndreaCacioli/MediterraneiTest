# MediterraneiTest

A sample webapp for review handling

# Environments (conda)

To export the environment do:

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

To import it automatically do:

```bash
conda env create -f environment.yml
```

To remove it do:

```bash
conda env remove --name mediterranei
```

