sft/trainer.py

<<<<<<< SEARCH
:start_line:30
:end_line:31
-------
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
=======
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        # Get validation repetitions from config
        self.val_repetitions = getattr(config, 'val_repetitions', 1)
        if self.val_repetitions > 10:
            print("Warning: Capping validation repetitions at 10")
            self.val_repetitions = 10
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:262
:end_line:265
-------
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
            
=======
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
        
        # Run validation multiple times if configured
        val_losses = []
        val_accs = []
        for _ in range(self.val_repetitions):
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:277
:end_line:281
-------
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        self.model.train()
        
        return val_loss, val_acc
=======
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Reset metric for next iteration
        val_metric.reset()
        
        # Average results across repetitions
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)
        
        self.model.train()
        return avg_val_loss, avg_val_acc
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:110
:end_line:125
-------
        # Create validation dataset if available
        try:
            val_dataset = PreformattedDataset(
                self.config.train_file_path,
                self.tokenizer,
                self.config.block_size,
                self.config.debug,
                split="validation"
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"No validation dataset found: {e}")
            val_dataset = None
            self.has_validation = False
=======
        # Load validation dataset from val_data.json
        val_data_path = os.path.join(os.path.dirname(self.config.train_file_path), 'val_data.json')
        if os.path.exists(val_data_path):
            logger.info(f"Loading validation data from {val_data_path}")
            with open(val_data_path, 'r') as f:
                val_data = json.load(f)
            val_dataset = PreformattedDataset(
                val_data,  # Pass the loaded validation data directly
                self.tokenizer,
                self.config.block_size,
                self.config.debug,
                is_eval_data=True  # Flag to handle eval-style formatting
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        else:
            logger.warning(f"No validation dataset found at {val_data_path}")
            val_dataset = None
            self.has_validation = False
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:262
:end_line:281
-------
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
            
        self.model.eval()
        val_metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                val_metric(outputs.logits, batch["labels"], loss)
        
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        self.model.train()
        
        return val_loss, val_acc
=======
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
            
        self.model.eval()
        val_metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                val_metric(outputs.logits, batch["labels"], loss)
                
                # Free up memory
                del outputs
                torch.cuda.empty_cache()
        
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        
        # Log validation sample count
        if self.accelerator.is_main_process:
            logger.info(f"Validated on {len(self.val_dataloader.dataset)} samples")
            
        self.model.train()
        return val_loss, val_acc
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:21
:end_line:21
-------
from .data_utils import PreformattedDataset
=======
from .data_utils import PreformattedDataset
from .eval_dataset import EvalDataset
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:110
:end_line:125
-------
        # Create validation dataset if available
        try:
            val_dataset = PreformattedDataset(
                self.config.train_file_path,
                self.tokenizer,
                self.config.block_size,
                self.config.debug,
                split="validation"
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"No validation dataset found: {e}")
            val_dataset = None
            self.has_validation = False
=======
        # Load validation dataset from val_data.json
        val_data_path = os.path.join(os.path.dirname(self.config.train_file_path), 'val_data.json')
        if os.path.exists(val_data_path):
            logger.info(f"Loading validation data from {val_data_path}")
            val_dataset = EvalDataset(
                val_data_path,
                self.tokenizer,
                self.config.block_size,
                self.config.debug
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        else:
            logger.warning(f"No validation dataset found at {val_data_path}")
            val_dataset = None
            self.has_validation = False
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:112
:end_line:131
-------
        # Load validation dataset from val_data.json
        val_data_path = os.path.join(os.path.dirname(self.config.train_file_path), 'val_data.json')
        if os.path.exists(val_data_path):
            logger.info(f"Loading validation data from {val_data_path}")
            with open(val_data_path, 'r') as f:
                val_data = json.load(f)
            val_dataset = PreformattedDataset(
                val_data,  # Pass the loaded validation data directly
                self.tokenizer,
                self.config.block_size,
                self.config.debug,
                is_eval_data=True  # Flag to handle eval-style formatting
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        else:
            logger.warning(f"No validation dataset found at {val_data_path}")
            val_dataset = None
            self.has_validation = False
            self.has_validation = False
=======
        # Load validation dataset from val_data.json
        val_data_path = os.path.join(os.path.dirname(self.config.train_file_path), 'val_data.json')
        if os.path.exists(val_data_path):
            logger.info(f"Loading validation data from {val_data_path}")
            val_dataset = EvalDataset(
                val_data_path,
                self.tokenizer,
                self.config.block_size,
                self.config.debug
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        else:
            logger.warning(f"No validation dataset found at {val_data_path}")
            val_dataset = None
            self.has_validation = False
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:22
:end_line:24
-------
from .data_utils import PreformattedDataset
from .eval_dataset import EvalDataset
from .metrics import SFTMetric
=======
from .data_utils import PreformattedDataset
from .val_dataset import ValDataset
from .metrics import SFTMetric
from .val_metrics import ValMetric
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:243
:end_line:245
-------
        # Initialize metric tracker with accelerator for consistent world size
        self.metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)

=======
        # Initialize metric trackers with accelerator for consistent world size
        self.train_metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)
        if self.has_validation:
            self.val_metric = ValMetric(
                device=self.accelerator.device,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator
            )

>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:263
:end_line:292
-------
    def validate(self):
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
            
        self.model.eval()
        val_metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                val_metric(outputs.logits, batch["labels"], loss)
                
                # Free up memory
                del outputs
                torch.cuda.empty_cache()
        
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        
        # Log validation sample count
        if self.accelerator.is_main_process:
            logger.info(f"Validated on {len(self.val_dataloader.dataset)} samples")
            
        self.model.train()
        return val_loss, val_acc
=======
    def validate(self):
        """Evaluate model on validation set.
        
        Computes both:
        1. Language modeling metrics (loss, token accuracy)
        2. Multiple choice metrics (answer accuracy per source)
        """
        if not self.has_validation:
            return 0.0, 0.0
            
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics with both loss and accuracy
                self.val_metric.update(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    loss=loss,
                    metadata=batch["metadata"]
                )
                
                # Free up memory
                del outputs
                torch.cuda.empty_cache()
        
        # Get validation metrics
        metrics, val_loss = self.val_metric.get_metric()
        
        # Use overall accuracy for early stopping
        val_acc = metrics['overall']['accuracy']
        
        # Log detailed metrics
        if self.accelerator.is_main_process:
            logger.info(f"\nValidation Metrics:")
            logger.info(f"  Language Modeling:")
            logger.info(f"    Loss: {val_loss:.4f}")
            logger.info(f"    Token Accuracy: {metrics['overall']['token_accuracy']:.4f}")
            logger.info(f"  Multiple Choice:")
            for source, source_metrics in metrics.items():
                if source != 'overall':
                    logger.info(f"    {source}:")
                    logger.info(f"      Accuracy: {source_metrics['accuracy']:.4f}")
                    logger.info(f"      Samples: {source_metrics['total_samples']}")
            logger.info(f"  Overall:")
            logger.info(f"    Accuracy: {val_acc:.4f}")
            logger.info(f"    Total Samples: {metrics['overall']['total_samples']}")
            
            # Log to wandb
            wandb.log({
                'val_loss': val_loss,
                'val_token_accuracy': metrics['overall']['token_accuracy'],
                'val_answer_accuracy': val_acc,
                **{f"val_{source}_accuracy": m['accuracy'] 
                   for source, m in metrics.items() if source != 'overall'}
            })
            
        self.model.train()
        return val_loss, val_acc
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:428
:end_line:429
-------
                    self.metric(outputs.logits, batch["labels"], loss)
                    acc, train_loss = self.metric.get_metric()
=======
                    self.train_metric(outputs.logits, batch["labels"], loss)
                    acc, train_loss = self.train_metric.get_metric()
>>>>>>> REPLACE

eval/eval.py

<<<<<<< SEARCH
:start_line:23
:end_line:33
-------
DEFAULT_REPETITIONS = {
    "GPQA_Medical_test": 5,  # Keep existing GPQA default
    "MedMCQA_validation": 1,
    "MedQA_USLME_test": 1,
    "PubMedQA_test": 1,
    "MMLU-Pro_Medical_test": 1,
    "MedDS": 1,
    "MedDS_NOTA": 1,
    "NEJMCRMC_qa": 1,
    "NEJMCRMC_mc": 1
}
=======
DEFAULT_REPETITIONS = {
    "GPQA_Medical_test": 1,
    "MedMCQA_validation": 1,
    "MedQA_USLME_test": 1,
    "PubMedQA_test": 1,
    "MMLU-Pro_Medical_test": 1,
    "MedDS": 1,
    "MedDS_NOTA": 1,
    "NEJMCRMC_qa": 1,
    "NEJMCRMC_mc": 1
}

# Default to 1 repetition, can be overridden in experiment config
DEFAULT_REPETITION_MULTIPLIER = 1
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:51
:end_line:61
-------
def get_repetitions_config(experiment_config=None):
    """Get repetitions config, merging defaults with experiment-specific config."""
    repetitions = DEFAULT_REPETITIONS.copy()
    
    if experiment_config and 'eval' in experiment_config:
        eval_config = experiment_config['eval']
        if 'repetitions' in eval_config:
            # Update defaults with experiment-specific values
            repetitions.update(eval_config['repetitions'])
    
    return repetitions
=======
def get_repetitions_config(experiment_config=None):
    """Get repetitions config, merging defaults with experiment-specific config."""
    repetitions = DEFAULT_REPETITIONS.copy()
    
    if experiment_config and 'eval' in experiment_config:
        eval_config = experiment_config['eval']
        
        # Get repetition multiplier (default to 1)
        multiplier = eval_config.get('repetition_multiplier', DEFAULT_REPETITION_MULTIPLIER)
        # Cap at 10
        multiplier = min(10, multiplier)
        
        # Apply multiplier to all sources
        if multiplier > 1:
            for source in repetitions:
                repetitions[source] = repetitions[source] * multiplier
        
        # Allow source-specific overrides
        if 'repetitions' in eval_config:
            repetitions.update(eval_config['repetitions'])
    
    return repetitions
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:140
:end_line:144
-------
def prepare_data(args, experiment):
    """Load and prepare data for evaluation."""
    print(f"\nLoading evaluation data with {args=}...")
    input_data = load_eval_dataset(args.experiment_name, args.path_to_eval_json)
    print(f"Loaded {len(input_data)} total examples")
=======
def prepare_data(args, experiment):
    """Load and prepare data for evaluation."""
    print(f"\nLoading evaluation data with {args=}...")
    input_data = load_eval_dataset(args.experiment_name, args.path_to_eval_json)
    print(f"Loaded {len(input_data)} total examples")
    
    # Load and exclude validation data if it exists
    val_data_path = os.path.join(os.path.dirname(args.path_to_eval_json), 'val_data.json')
    if os.path.exists(val_data_path):
        print(f"Loading validation data from {val_data_path}")
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
            
        # Create set of validation sample IDs for efficient lookup
        val_ids = {(item.get('source', ''), item.get('id', '')) for item in val_data}
        
        # Filter out validation samples
        input_data = [
            item for item in input_data 
            if (item.get('source', ''), item.get('id', '')) not in val_ids
        ]
        print(f"Excluded {len(val_data)} validation samples")
        print(f"Using {len(input_data)} examples for evaluation")
>>>>>>> REPLACE

