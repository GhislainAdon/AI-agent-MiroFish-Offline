import service, { requestWithRetry } from './index'

/**
 * Générer l'ontologie (télécharger les documents et les exigences de simulation)
 * @param {Object} data - Contient les fichiers, simulation_requirement, project_name, etc.
 * @returns {Promise}
 */
export function generateOntology(formData) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/ontology/generate',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  )
}

/**
 * Construire le graphe
 * @param {Object} data - Contient project_id, graph_name, etc.
 * @returns {Promise}
 */
export function buildGraph(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/build',
      method: 'post',
      data
    })
  )
}

/**
 * Consulter le statut de la tâche
 * @param {String} taskId - Identifiant de la tâche
 * @returns {Promise}
 */
export function getTaskStatus(taskId) {
  return service({
    url: `/api/graph/task/${taskId}`,
    method: 'get'
  })
}

/**
 * Récupérer les données du graphe
 * @param {String} graphId - Identifiant du graphe
 * @returns {Promise}
 */
export function getGraphData(graphId) {
  return service({
    url: `/api/graph/data/${graphId}`,
    method: 'get'
  })
}

/**
 * Récupérer les informations du projet
 * @param {String} projectId - Identifiant du projet
 * @returns {Promise}
 */
export function getProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}`,
    method: 'get'
  })
}
