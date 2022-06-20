from clhive import(
    config_parser,
    get_loaders_and_transforms,
    get_method,
    evaluators,
    Trainer
)


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="tmp"
    )

    loaders = get_loaders_and_transforms(config=config)

    method = get_method(config=config, model=)

    probe_evaluator = evaluators.ProbeEvaluator(
        method,
        train_loader=loaders.train_loader, 
        test_loader=loaders.test_loader, 
        config=config, 
        logger=None
    )

    trainer = Trainer(method, train_loader=loaders.train_loader, config=config)

    trainer.fit()

if __name__ == '__main__':
    main()