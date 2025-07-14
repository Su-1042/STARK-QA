import os
import json
import time
from typing import Dict, List, Any, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import textstat
except ImportError:
    # Will be installed via conda environment
    pass

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for QA systems."""
    
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluation tools
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            
            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
            
            # BLEU smoothing function
            self.smoothie = SmoothingFunction().method4
            
        except Exception as e:
            self.logger.warning(f"Some evaluation metrics may not be available: {e}")
    
    def evaluate_system(self, system, test_data: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
        """Evaluate a QA system on test data."""
        self.logger.info(f"Evaluating {system_name}...")
        
        results = {
            'system_name': system_name,
            'predictions': [],
            'metrics': {},
            'timing': {}
        }
        
        # Track timing
        start_time = time.time()
        answer_times = []
        
        for i, item in enumerate(test_data):
            question = item['question']
            ground_truth = item['answer']
            
            # Time individual predictions
            pred_start = time.time()
            
            try:
                if hasattr(system, 'answer_question'):
                    if system_name == "STARK-QA":
                        # STARK-QA returns a dict
                        response = system.answer_question(question)
                        prediction = response['answer'] if isinstance(response, dict) else str(response)
                    else:
                        # Baselines return string
                        prediction = system.answer_question(question)
                else:
                    prediction = "System error: answer_question method not found"
            except Exception as e:
                self.logger.error(f"Error getting prediction for question {i}: {e}")
                prediction = "System error during prediction"
            
            pred_time = time.time() - pred_start
            answer_times.append(pred_time)
            
            # Store result
            result_item = {
                'question_id': item.get('id', i),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'question_type': item.get('attribute', 'unknown'),
                'local_or_sum': item.get('local_or_sum', 'unknown'),
                'ex_or_im': item.get('ex_or_im', 'unknown'),
                'prediction_time': pred_time
            }
            
            results['predictions'].append(result_item)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(test_data)} questions")
        
        # Calculate timing metrics
        total_time = time.time() - start_time
        results['timing'] = {
            'total_time': total_time,
            'avg_time_per_question': total_time / len(test_data),
            'median_time_per_question': sorted(answer_times)[len(answer_times)//2]
        }
        
        # Calculate evaluation metrics
        results['metrics'] = self._calculate_metrics(results['predictions'])
        
        # Save detailed results
        self._save_results(results, system_name)
        
        return results
    
    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Extract predictions and ground truths
        preds = [item['prediction'] for item in predictions]
        truths = [item['ground_truth'] for item in predictions]
        
        # Basic metrics
        metrics['total_questions'] = len(predictions)
        metrics['valid_predictions'] = sum(1 for p in preds if p and not p.startswith('System error'))
        
        # ROUGE scores
        try:
            rouge_scores = self._calculate_rouge_scores(preds, truths)
            metrics.update(rouge_scores)
        except Exception as e:
            self.logger.warning(f"Could not calculate ROUGE scores: {e}")
        
        # BLEU scores
        try:
            bleu_scores = self._calculate_bleu_scores(preds, truths)
            metrics.update(bleu_scores)
        except Exception as e:
            self.logger.warning(f"Could not calculate BLEU scores: {e}")
        
        # BERTScore
        try:
            bert_scores = self._calculate_bert_scores(preds, truths)
            metrics.update(bert_scores)
        except Exception as e:
            self.logger.warning(f"Could not calculate BERTScore: {e}")
        
        # Length metrics
        pred_lengths = [len(p.split()) for p in preds if p]
        truth_lengths = [len(t.split()) for t in truths if t]
        
        metrics['avg_prediction_length'] = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        metrics['avg_ground_truth_length'] = sum(truth_lengths) / len(truth_lengths) if truth_lengths else 0
        
        # Readability metrics
        try:
            readability_scores = self._calculate_readability_scores(preds)
            metrics.update(readability_scores)
        except Exception as e:
            self.logger.warning(f"Could not calculate readability scores: {e}")
        
        # Question type breakdown
        type_metrics = self._calculate_type_metrics(predictions)
        metrics['by_question_type'] = type_metrics
        
        return metrics
    
    def _calculate_rouge_scores(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, truth in zip(predictions, ground_truths):
            if pred and truth:
                scores = self.rouge_scorer.score(truth, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1_avg': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            'rouge2_avg': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            'rougeL_avg': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
        }
    
    def _calculate_bleu_scores(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores."""
        bleu_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            if pred and truth:
                pred_tokens = pred.lower().split()
                truth_tokens = [truth.lower().split()]  # BLEU expects list of reference lists
                
                try:
                    bleu = sentence_bleu(truth_tokens, pred_tokens, smoothing_function=self.smoothie)
                    bleu_scores.append(bleu)
                except:
                    continue
        
        return {
            'bleu_avg': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        }
    
    def _calculate_bert_scores(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        valid_preds = []
        valid_truths = []
        
        for pred, truth in zip(predictions, ground_truths):
            if pred and truth and not pred.startswith('System error'):
                valid_preds.append(pred)
                valid_truths.append(truth)
        
        if not valid_preds:
            return {'bert_score_f1': 0.0}
        
        P, R, F1 = bert_score(valid_preds, valid_truths, lang="en", verbose=False)
        
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }
    
    def _calculate_readability_scores(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate readability metrics."""
        valid_preds = [p for p in predictions if p and not p.startswith('System error')]
        
        if not valid_preds:
            return {}
        
        flesch_scores = []
        gunning_fog_scores = []
        
        for pred in valid_preds:
            try:
                flesch_scores.append(textstat.flesch_reading_ease(pred))
                gunning_fog_scores.append(textstat.gunning_fog(pred))
            except:
                continue
        
        return {
            'flesch_reading_ease': sum(flesch_scores) / len(flesch_scores) if flesch_scores else 0,
            'gunning_fog_index': sum(gunning_fog_scores) / len(gunning_fog_scores) if gunning_fog_scores else 0
        }
    
    def _calculate_type_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics by question type."""
        type_groups = defaultdict(list)
        
        for item in predictions:
            q_type = item['question_type']
            local_sum = item['local_or_sum']
            ex_im = item['ex_or_im']
            
            type_groups[q_type].append(item)
            type_groups[f"local_sum_{local_sum}"].append(item)
            type_groups[f"ex_im_{ex_im}"].append(item)
        
        type_metrics = {}
        for group_name, items in type_groups.items():
            if len(items) > 0:
                valid_preds = sum(1 for item in items 
                                if item['prediction'] and not item['prediction'].startswith('System error'))
                type_metrics[group_name] = {
                    'count': len(items),
                    'valid_predictions': valid_preds,
                    'success_rate': valid_preds / len(items)
                }
        
        return type_metrics
    
    def _save_results(self, results: Dict[str, Any], system_name: str):
        """Save detailed results to files."""
        # Save JSON results
        json_file = os.path.join(self.output_dir, f"{system_name}_results.json")
        with open(json_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = os.path.join(self.output_dir, f"{system_name}_predictions.csv")
        df = pd.DataFrame(results['predictions'])
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Saved {system_name} results to {json_file} and {csv_file}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def generate_comparison_report(self, system_results: List[Tuple[str, Dict[str, Any]]]):
        """Generate comprehensive comparison report."""
        self.logger.info("Generating comparison report...")
        
        # Create comparison dataframe
        comparison_data = []
        for system_name, results in system_results:
            metrics = results['metrics']
            timing = results['timing']
            
            row = {
                'System': system_name,
                'Total Questions': metrics.get('total_questions', 0),
                'Valid Predictions': metrics.get('valid_predictions', 0),
                'Success Rate': metrics.get('valid_predictions', 0) / metrics.get('total_questions', 1),
                'ROUGE-1': metrics.get('rouge1_avg', 0),
                'ROUGE-2': metrics.get('rouge2_avg', 0),
                'ROUGE-L': metrics.get('rougeL_avg', 0),
                'BLEU': metrics.get('bleu_avg', 0),
                'BERTScore F1': metrics.get('bert_score_f1', 0),
                'Avg Time (s)': timing.get('avg_time_per_question', 0),
                'Total Time (s)': timing.get('total_time', 0)
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_file = os.path.join(self.output_dir, "system_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        # Generate plots
        self._generate_comparison_plots(comparison_df, system_results)
        
        # Generate detailed report
        self._generate_detailed_report(comparison_df, system_results)
        
        self.logger.info(f"Comparison report saved to {self.output_dir}")
    
    def _generate_comparison_plots(self, comparison_df: pd.DataFrame, system_results: List[Tuple[str, Dict[str, Any]]]):
        """Generate comparison plots."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('STARK-QA System Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: ROUGE scores
        rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        rouge_data = comparison_df[['System'] + rouge_metrics].set_index('System')
        rouge_data.plot(kind='bar', ax=axes[0, 0], title='ROUGE Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(title='Metrics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Overall performance
        performance_metrics = ['Success Rate', 'BLEU', 'BERTScore F1']
        performance_data = comparison_df[['System'] + performance_metrics].set_index('System')
        performance_data.plot(kind='bar', ax=axes[0, 1], title='Overall Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend(title='Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Timing comparison (logarithmic scale)
        timing_data = comparison_df[['System', 'Avg Time (s)']].set_index('System')
        timing_data.plot(kind='bar', ax=axes[1, 0], title='Average Response Time (Log Scale)', color='orange')
        axes[1, 0].set_ylabel('Time (seconds) - Log Scale')
        axes[1, 0].set_yscale('log')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Question type performance (if available)
        if len(system_results) > 0:
            # Use first system's type breakdown as example
            _, first_results = system_results[0]
            type_metrics = first_results['metrics'].get('by_question_type', {})
            
            if type_metrics:
                type_names = []
                success_rates = []
                
                for type_name, metrics in type_metrics.items():
                    if 'success_rate' in metrics and not type_name.startswith('local_sum') and not type_name.startswith('ex_im'):
                        type_names.append(type_name)
                        success_rates.append(metrics['success_rate'])
                
                if type_names:
                    axes[1, 1].bar(type_names, success_rates, color='green', alpha=0.7)
                    axes[1, 1].set_title('Success Rate by Question Type (STARK-QA)')
                    axes[1, 1].set_ylabel('Success Rate')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No question type data available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, 'No question type data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "system_comparison_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_detailed_report(self, comparison_df: pd.DataFrame, system_results: List[Tuple[str, Dict[str, Any]]]):
        """Generate detailed text report."""
        report_file = os.path.join(self.output_dir, "evaluation_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# STARK-QA Evaluation Report\n\n")
            f.write("## System Comparison Overview\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Detailed Analysis\n\n")
            
            # Find best performing system
            best_system = comparison_df.loc[comparison_df['BERTScore F1'].idxmax(), 'System']
            f.write(f"**Best Overall System:** {best_system}\n\n")
            
            # Performance analysis
            for system_name, results in system_results:
                f.write(f"### {system_name}\n\n")
                
                metrics = results['metrics']
                timing = results['timing']
                
                f.write(f"- **Total Questions Processed:** {metrics.get('total_questions', 0)}\n")
                f.write(f"- **Valid Predictions:** {metrics.get('valid_predictions', 0)}\n")
                f.write(f"- **Success Rate:** {metrics.get('valid_predictions', 0) / metrics.get('total_questions', 1):.3f}\n")
                f.write(f"- **Average Response Time:** {timing.get('avg_time_per_question', 0):.3f} seconds\n")
                f.write(f"- **ROUGE-1 Score:** {metrics.get('rouge1_avg', 0):.3f}\n")
                f.write(f"- **ROUGE-L Score:** {metrics.get('rougeL_avg', 0):.3f}\n")
                f.write(f"- **BLEU Score:** {metrics.get('bleu_avg', 0):.3f}\n")
                f.write(f"- **BERTScore F1:** {metrics.get('bert_score_f1', 0):.3f}\n\n")
                
                # Question type breakdown
                type_metrics = metrics.get('by_question_type', {})
                if type_metrics:
                    f.write("**Performance by Question Type:**\n\n")
                    for type_name, type_data in type_metrics.items():
                        if not type_name.startswith('local_sum') and not type_name.startswith('ex_im'):
                            f.write(f"- {type_name}: {type_data['success_rate']:.3f} ({type_data['valid_predictions']}/{type_data['count']})\n")
                    f.write("\n")
            
            f.write("## Conclusions\n\n")
            f.write("This evaluation compares STARK-QA (combining RAG and KAG) against basic RAG and KAG baselines.\n")
            f.write("The metrics include semantic similarity (ROUGE, BLEU, BERTScore), response time, and question type analysis.\n")
        
        self.logger.info(f"Detailed report saved to {report_file}")