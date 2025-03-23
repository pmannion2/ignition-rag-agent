// Cursor Extension for Ignition RAG
// This extension integrates the Ignition RAG system with Cursor IDE

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Configuration
const DEFAULT_RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8001';
const DEFAULT_PYTHON_SCRIPT = path.join(__dirname, 'cursor_client.py');
const SHELL_SCRIPT_WRAPPER = path.join(__dirname, 'run_client.sh');

// Extension configuration
let config = {
  ragApiUrl: process.env.RAG_API_URL || DEFAULT_RAG_API_URL,
  pythonPath: process.env.PYTHON_PATH || 'python3',
  clientScript: fs.existsSync(SHELL_SCRIPT_WRAPPER) ? SHELL_SCRIPT_WRAPPER : DEFAULT_PYTHON_SCRIPT,
  enabled: true,
  topK: 3,
};

// Check if the client script exists
function verifySetup() {
  try {
    if (!fs.existsSync(config.clientScript)) {
      console.error(`Cursor client script not found at: ${config.clientScript}`);
      return false;
    }
    return true;
  } catch (error) {
    console.error(`Error verifying setup: ${error.message}`);
    return false;
  }
}

/**
 * Get context from the Ignition RAG system
 * 
 * @param {string} query - The query to search for
 * @param {Object} context - Editor context (current file, etc.)
 * @returns {Promise<string>} Context from the RAG system
 */
async function getIgnitionContext(query, context = {}) {
  return new Promise((resolve, reject) => {
    try {
      // Verify setup
      if (!verifySetup()) {
        return resolve(`Error: Ignition RAG client setup is incomplete or invalid.`);
      }

      // Skip if disabled
      if (!config.enabled) {
        return resolve('');
      }

      // Get current file path from cursor context
      const currentFile = context.currentFile || '';

      // Determine if we're using the shell script wrapper
      const isUsingWrapper = config.clientScript.endsWith('run_client.sh');
      
      // Prepare arguments for Python script or shell wrapper
      let args = [];
      if (isUsingWrapper) {
        // Shell script already includes the path to the Python script
        args = [
          query,
          '--file', currentFile,
          '--top-k', config.topK.toString(),
          '--output', 'text'
        ];
      } else {
        // Using Python directly
        args = [
          config.clientScript,
          query,
          '--file', currentFile,
          '--top-k', config.topK.toString(),
          '--output', 'text'
        ];
      }

      // Prepare the command to run
      const command = isUsingWrapper ? config.clientScript : config.pythonPath;
      
      // Spawn process
      const process = spawn(command, args, {
        env: { ...process.env, RAG_API_URL: config.ragApiUrl }
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code !== 0) {
          console.error(`Ignition RAG client exited with code ${code}`);
          console.error(`Error: ${stderr}`);
          return resolve(`Error retrieving Ignition RAG context: ${stderr}`);
        }
        resolve(stdout);
      });
    } catch (error) {
      console.error(`Error in getIgnitionContext: ${error.message}`);
      resolve(`Error: ${error.message}`);
    }
  });
}

/**
 * Enhance a Cursor prompt with Ignition RAG context
 * 
 * @param {string} prompt - The original prompt
 * @param {Object} context - Editor context (current file, etc.)
 * @returns {Promise<string>} Enhanced prompt with context
 */
async function enhancePrompt(prompt, context = {}) {
  try {
    // Skip enhancement for non-ignition related prompts
    const ignitionKeywords = ['ignition', 'perspective', 'tag', 'view', 'project', 'component'];
    const shouldEnhance = ignitionKeywords.some(keyword => prompt.toLowerCase().includes(keyword));
    
    if (!shouldEnhance) {
      return prompt;
    }
    
    // Get context from RAG system
    const ragContext = await getIgnitionContext(prompt, context);
    
    // Only enhance if we got meaningful context
    if (ragContext && !ragContext.startsWith('Error:') && !ragContext.includes('No relevant context found')) {
      return `${prompt}\n\n${ragContext}`;
    }
    
    return prompt;
  } catch (error) {
    console.error(`Error enhancing prompt: ${error.message}`);
    return prompt;
  }
}

// Extension API
module.exports = {
  getIgnitionContext,
  enhancePrompt,
  
  // Configuration API
  configure: (options = {}) => {
    config = { ...config, ...options };
    return config;
  },
  
  // Enable/disable the extension
  enable: () => {
    config.enabled = true;
    return true;
  },
  
  disable: () => {
    config.enabled = false;
    return true;
  },
  
  // Cursor integration hooks
  hooks: {
    // Hook for enhancing prompt completions
    beforePromptCompletion: async (params) => {
      const { prompt, file } = params;
      const enhancedPrompt = await enhancePrompt(prompt, { currentFile: file });
      return { ...params, prompt: enhancedPrompt };
    }
  }
}; 