"""
Minimal confmap implementation for configuration file conversion
This avoids the heavy dependencies of the original confmap library
"""
import json
import yaml
import configparser
from pathlib import Path

def from_file(file_path, to='yaml'):
    """Convert configuration file to specified format"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = path.read_text(encoding='utf-8')
    return from_string(content, to, path.suffix.lower())

def from_string(content, to='yaml', from_format=None):
    """Convert configuration string to specified format"""
    # Detect input format
    if from_format is None:
        from_format = detect_format(content)
    
    # Parse input
    data = parse_config(content, from_format)
    
    # Convert to output format
    return convert_to_format(data, to)

def detect_format(content):
    """Detect configuration format"""
    content = content.strip()
    
    if content.startswith('{') or content.startswith('['):
        return '.json'
    elif content.startswith('---') or ': ' in content.split('\n')[0]:
        return '.yaml'
    elif '=' in content.split('\n')[0] and '[' not in content.split('\n')[0]:
        return '.ini'
    else:
        # Default to trying YAML
        return '.yaml'

def parse_config(content, format_type):
    """Parse configuration based on format"""
    try:
        if format_type in ['.json']:
            return json.loads(content)
        elif format_type in ['.yaml', '.yml']:
            return yaml.safe_load(content)
        elif format_type in ['.ini', '.cfg', '.conf']:
            config = configparser.ConfigParser()
            config.read_string(content)
            # Convert to dict
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            # Add DEFAULT section if present
            if config.defaults():
                result['DEFAULT'] = dict(config.defaults())
            return result
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        raise ValueError(f"Failed to parse {format_type} content: {str(e)}")

def convert_to_format(data, to_format):
    """Convert data to specified format"""
    if to_format == 'json':
        return json.dumps(data, indent=2, ensure_ascii=False)
    elif to_format in ['yaml', 'yml']:
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    elif to_format == 'ini':
        return dict_to_ini(data)
    else:
        raise ValueError(f"Unsupported output format: {to_format}")

def dict_to_ini(data):
    """Convert dictionary to INI format"""
    config = configparser.ConfigParser()
    
    for key, value in data.items():
        if isinstance(value, dict):
            # This is a section
            if key == 'DEFAULT':
                continue
            config.add_section(key)
            for subkey, subvalue in value.items():
                config.set(key, subkey, str(subvalue))
        else:
            # This goes in DEFAULT section
            if not config.has_section('DEFAULT'):
                config.add_section('DEFAULT')
            config.set('DEFAULT', key, str(value))
    
    # Write to string
    import io
    output = io.StringIO()
    config.write(output)
    return output.getvalue()

# Alias functions to match original confmap API
load = from_file
loads = from_string
