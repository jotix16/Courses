// Auto-generated. Do not edit!

// (in-package svea.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class average_velocity {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.average_velocity = null;
    }
    else {
      if (initObj.hasOwnProperty('average_velocity')) {
        this.average_velocity = initObj.average_velocity
      }
      else {
        this.average_velocity = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type average_velocity
    // Serialize message field [average_velocity]
    bufferOffset = _serializer.float64(obj.average_velocity, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type average_velocity
    let len;
    let data = new average_velocity(null);
    // Deserialize message field [average_velocity]
    data.average_velocity = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'svea/average_velocity';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '320db02e2aa0448db8c012457af99cee';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 average_velocity
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new average_velocity(null);
    if (msg.average_velocity !== undefined) {
      resolved.average_velocity = msg.average_velocity;
    }
    else {
      resolved.average_velocity = 0.0
    }

    return resolved;
    }
};

module.exports = average_velocity;
