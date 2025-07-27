# Auto-Resume System Redesign Plan

## Current Problems

### 1. Configuration Ignored
- `auto_resume: false` in best_strategy.json is ignored
- Server forces auto-resume regardless of config
- No way to disable it without removing resume file

### 2. Data Format Confusion
- LONG positions need BTC amounts, SHORT needs USD amounts
- Resume file doesn't validate this
- Error messages are unclear: "USD amount requires 'short' position"

### 3. Strategy-Specific Issues
- MACrossoverStrategy saves as "AdaptiveMultiStrategy"
- Different strategies have different resume requirements
- No validation that resume data matches current strategy

### 4. Missing Attributes
- `last_trade_time` not initialized in base strategy
- Different strategies expect different attributes
- Crashes when saving resume state

### 5. Poor Error Handling
- Silent failures when resume file can't be saved
- Cryptic error messages during resume
- No fallback when resume fails

## Proposed Solution

### Phase 1: Clean Configuration System

```python
# In tdr_server.py
def should_auto_resume(config):
    """Determine if auto-resume should run."""
    # 1. Check command line args first
    if args.no_auto_resume:
        return False
    
    # 2. Check best_strategy.json
    if config.get('best_strategy', {}).get('auto_resume') == False:
        return False
    
    # 3. Check if resume file exists
    if not os.path.exists('resume-auto-trade.json'):
        return False
        
    return True
```

### Phase 2: Strategy-Aware Resume System

```python
class BaseStrategy:
    """Base class with proper resume support."""
    
    def __init__(self, ...):
        # ALWAYS initialize these
        self.last_trade_time = None
        self.position = 0
        self.position_size = 0.0
        self.position_cost_basis = 0.0
        
    @classmethod
    def get_resume_schema(cls):
        """Return required fields for this strategy."""
        return {
            'required': ['position', 'amount', 'unit', 'entry_price'],
            'strategy_type': cls.__name__,
            'validation_rules': cls.get_validation_rules()
        }
    
    @classmethod
    def validate_resume_data(cls, data):
        """Validate resume data for this strategy."""
        errors = []
        
        # Check required fields
        schema = cls.get_resume_schema()
        for field in schema['required']:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate position/amount consistency
        if 'position' in data and 'unit' in data:
            position = data['position'].upper()
            unit = data['unit'].lower()
            
            if position == 'LONG' and unit != 'btc':
                errors.append("LONG positions must use BTC amounts")
            elif position == 'SHORT' and unit != 'usd':
                errors.append("SHORT positions must use USD amounts")
        
        return errors

class MACrossoverStrategy(BaseStrategy):
    """MA strategy with proper resume support."""
    
    @classmethod
    def get_validation_rules(cls):
        return {
            'position': ['LONG', 'SHORT'],
            'unit': ['btc', 'usd'],
            'amount': lambda x: x > 0,
            'entry_price': lambda x: x > 0
        }
    
    def save_resume_state(self):
        """Save with proper error handling and validation."""
        try:
            # Build resume data
            resume_data = self._build_resume_data()
            
            # Validate before saving
            errors = self.validate_resume_data(resume_data)
            if errors:
                self.logger.error(f"Resume validation failed: {errors}")
                return False
            
            # Save with atomic write
            resume_file = 'resume-auto-trade.json'
            temp_file = resume_file + '.tmp'
            
            with open(temp_file, 'w') as f:
                json.dump(resume_data, f, indent=2)
            
            # Atomic rename
            os.rename(temp_file, resume_file)
            
            self.logger.info(f"✅ Resume state saved: {resume_data['position']} "
                           f"{resume_data['amount']}{resume_data['unit']} @ ${resume_data['entry_price']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save resume state: {e}")
            return False
    
    def _build_resume_data(self):
        """Build resume data with correct format."""
        if self.position == 1:  # LONG
            return {
                'timestamp': datetime.now().isoformat(),
                'position': 'LONG',
                'amount': round(self.balance_btc, 8),
                'unit': 'btc',
                'entry_price': self._calculate_entry_price(),
                'strategy_type': self.__class__.__name__,
                'strategy_params': {
                    'short_window': self.short_window,
                    'long_window': self.long_window
                }
            }
        elif self.position == -1:  # SHORT
            return {
                'timestamp': datetime.now().isoformat(),
                'position': 'SHORT',
                'amount': round(self.balance_usd, 2),
                'unit': 'usd',
                'entry_price': self._calculate_entry_price(),
                'strategy_type': self.__class__.__name__,
                'strategy_params': {
                    'short_window': self.short_window,
                    'long_window': self.long_window
                }
            }
        else:
            raise ValueError("Cannot save resume state for neutral position")
```

### Phase 3: Improved Auto-Resume Flow

```python
# In tdr_server.py
def auto_resume_with_validation(shell, resume_file):
    """Auto-resume with proper validation and error handling."""
    
    try:
        # 1. Load resume data
        with open(resume_file, 'r') as f:
            resume_data = json.load(f)
        
        logger.info(f"Found resume file: {resume_data.get('position', 'UNKNOWN')} "
                   f"{resume_data.get('amount', 0)}{resume_data.get('unit', '?')}")
        
        # 2. Validate strategy match
        current_strategy = shell.config.get('strategy_type', 'MA')
        resume_strategy = resume_data.get('strategy_type', 'MACrossoverStrategy')
        
        if not strategies_compatible(current_strategy, resume_strategy):
            logger.warning(f"Strategy mismatch: current={current_strategy}, "
                         f"resume={resume_strategy}")
            return False
        
        # 3. Get strategy class and validate data
        strategy_class = get_strategy_class(current_strategy)
        errors = strategy_class.validate_resume_data(resume_data)
        
        if errors:
            logger.error(f"Resume validation failed: {errors}")
            return False
        
        # 4. Build auto_trade command
        position = resume_data['position'].lower()
        amount = resume_data['amount']
        unit = resume_data['unit']
        entry_price = resume_data.get('entry_price', 0)
        
        # Build command based on strategy
        if current_strategy == 'MA':
            params = resume_data.get('strategy_params', {})
            short = params.get('short_window', 4)
            long = params.get('long_window', 20)
            
            cmd = f"auto_trade {amount}{unit} MA short={short} long={long} "
            cmd += f"do_live_trades=True hist_position={position}"
            
            if entry_price > 0:
                cmd += f" entry_price={entry_price}"
        else:
            # Other strategies...
            pass
        
        # 5. Execute with timeout
        logger.info(f"Executing auto-resume: {cmd}")
        
        result = shell.onecmd_with_timeout(cmd, timeout=10)
        
        if "Auto-trading started" in result:
            logger.info("✅ Auto-resume successful")
            return True
        else:
            logger.error(f"Auto-resume failed: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Auto-resume error: {e}")
        return False
```

### Phase 4: Testing Framework

```python
# tests/test_auto_resume.py
class TestAutoResume(unittest.TestCase):
    
    def test_long_position_resume(self):
        """Test resuming a LONG position."""
        resume_data = {
            'position': 'LONG',
            'amount': 1.5,
            'unit': 'btc',
            'entry_price': 100000,
            'strategy_type': 'MACrossoverStrategy'
        }
        
        errors = MACrossoverStrategy.validate_resume_data(resume_data)
        self.assertEqual(errors, [])
    
    def test_invalid_long_with_usd(self):
        """Test that LONG with USD is rejected."""
        resume_data = {
            'position': 'LONG',
            'amount': 150000,
            'unit': 'usd',  # Wrong!
            'entry_price': 100000,
            'strategy_type': 'MACrossoverStrategy'
        }
        
        errors = MACrossoverStrategy.validate_resume_data(resume_data)
        self.assertIn("LONG positions must use BTC amounts", errors)
    
    def test_auto_resume_disabled(self):
        """Test that auto_resume:false prevents resume."""
        config = {'best_strategy': {'auto_resume': False}}
        self.assertFalse(should_auto_resume(config))
```

### Phase 5: Migration Path

1. **Add deprecation warnings** to current system
2. **Run both systems in parallel** for testing
3. **Add feature flag** to enable new system
4. **Gradual rollout** with monitoring
5. **Remove old system** after validation

### Implementation Priority

1. **Critical**: Fix configuration check (auto_resume: false)
2. **High**: Add validation for position/amount consistency  
3. **High**: Fix strategy type in resume files
4. **Medium**: Add comprehensive error messages
5. **Low**: Add testing framework

### Success Metrics

- Zero resume failures due to validation errors
- Clear error messages when resume fails
- Ability to disable auto-resume via config
- All strategies save correct resume format
- 100% test coverage for resume scenarios

This plan addresses all current bugs and provides a robust foundation for the auto-resume system.