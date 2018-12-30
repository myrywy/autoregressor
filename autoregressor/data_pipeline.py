class DataPipeline:
    def __init__(self):
        self._transformations = []


    def add_unit_transformation(
            self,
            transformation_funtion,
            *keys
            ):
        """Adds transformation of some unit in example to pipeline. Access unit in example by chain of keys used as example[keys[0]][keys[1]]...[keys[n]]

        Warning:
            If example yielded by dataset iterator has a form of one-element tuple then it will be reduced to this
            element - without enclosing tuple. This is because of the way Dataset.map function works.

        Args:
            transformation_funtion (Function[(tf.Tensor,), tf.Tensor]): function that takes tensor and return tensor, 
            keys (Iterable): keys used one by one to access element that is to be transformed in example
        
        Returns:
            None
        """
        def get_unit(example):
            for key in keys:
                example = example[key]
            return example

        def set_unit(unit, example, keys=keys):
            if not keys:
                return unit
            current_level = example
            key = keys[0]
            if isinstance(example, tuple):
                current_level = list(current_level)
                current_level[key] = set_unit(unit, current_level[key], keys=keys[1:])
                current_level = tuple(current_level)
            elif isinstance(example, dict):
                current_level = current_level.copy()
                current_level[key] = set_unit(unit, current_level[key], keys=keys[1:])
            else:
                raise ValueError("Unsupported container. Only dict and tuple allowed.")
            return current_level

        self.add_generic_unit_transformation_in_nested_structure(transformation_funtion, get_unit, set_unit)

    def add_generic_unit_transformation_in_nested_structure(
            self,
            transformation_funtion, 
            get_unit_from_example_function, 
            set_unit_in_example_function,
            ):
        """Adds transformation of some unit in example to pipeline.
        
        Args:
            transformation_funtion (Function[(tf.Tensor,), tf.Tensor]): function that takes tensor and return tensor, 
            get_unit_from_example_function: function that takes tuple of elements in dataset's example (for exaple tuple(features, labels)) and returns element that is to be transformed, 
            set_unit_in_example_function: function of signature: (unit, example) -> example that replace some element in example (that is some tuple of possibly nested dicts) with unit

        Returns:
            None
        """
        def example_transformation(*args):
            if len(args) == 1:
                args = args[0]
            else:
                args = args
            unit = get_unit_from_example_function(args)
            result = transformation_funtion(unit)
            args = set_unit_in_example_function(result, args)
            return args

        self._transformations.append(example_transformation)
    
    def add_structural_transformation(
            self,
            transformation_function,
            ):
        """Adds transformation of example to data pipeline

        Args:
            transformation_function: function that modifies dataset's example
        """
        self._transformations.append(transformation_function)

    def transform_dataset(self, dataset):
        """Sequentially applies added transformations to dataset.

        Args:
            dataset (tf.data.Dataset): a dataset which elements are to be transformed

        Returns:
            tf.data.Dataset: transformed dataset
        """
        for transformation in self._transformations:
            dataset = dataset.map(transformation)
        return dataset

