import service, { requestWithRetry } from './index'

/**
 * Démarrer la génération du rapport
 * @param {Object} data - { simulation_id, force_regenerate? }
 */
export const generateReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/generate', data), 3, 1000)
}

/**
 * Récupérer le statut de génération du rapport
 * @param {string} reportId
 */
export const getReportStatus = (reportId) => {
  return service.get(`/api/report/generate/status`, { params: { report_id: reportId } })
}

/**
 * Récupérer le journal de l'agent (incrémental)
 * @param {string} reportId
 * @param {number} fromLine - Commencer à partir de cette ligne
 */
export const getAgentLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/agent-log`, { params: { from_line: fromLine } })
}

/**
 * Récupérer le journal de la console (incrémental)
 * @param {string} reportId
 * @param {number} fromLine - Commencer à partir de cette ligne
 */
export const getConsoleLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/console-log`, { params: { from_line: fromLine } })
}

/**
 * Récupérer les détails du rapport
 * @param {string} reportId
 */
export const getReport = (reportId) => {
  return service.get(`/api/report/${reportId}`)
}

/**
 * Dialoguer avec ReportAgent.
 * @param {Object} data - { simulation_id, message, chat_history? }
 */
export const chatWithReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/chat', data), 3, 1000)
}
